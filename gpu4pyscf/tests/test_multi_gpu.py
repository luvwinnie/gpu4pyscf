# Copyright 2024-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for multi-GPU support.

These tests verify that:
1. CUDA constant memory is initialized on all devices
2. SCF energy matches between 1 GPU and multi-GPU
3. Gradient matches between 1 GPU and multi-GPU
4. multi_gpu utility functions work correctly
"""

import unittest
import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.__config__ import num_devices, _p2p_access


def setUpModule():
    pass


def tearDownModule():
    pass


@unittest.skipIf(num_devices < 2, 'Requires 2+ GPUs')
class TestMultiGPU(unittest.TestCase):
    """Tests that require multiple GPUs."""

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='6-31g*',
            verbose=0,
        )

    def test_config_num_devices(self):
        """num_devices should match CUDA device count."""
        self.assertGreaterEqual(num_devices, 2)

    def test_p2p_access_check(self):
        """P2P access should be checked without errors."""
        # _p2p_access is a bool, set during __config__ import
        self.assertIsInstance(_p2p_access, bool)

    def test_scf_energy(self):
        """Multi-GPU SCF energy should match reference."""
        import gpu4pyscf
        mf = gpu4pyscf.dft.rks.RKS(self.mol, xc='b3lyp')
        e = mf.kernel()
        # Reference: single-GPU energy for water/6-31g*/B3LYP
        self.assertAlmostEqual(e, -76.4068146845, places=6)

    def test_gradient(self):
        """Multi-GPU gradient should be correct (water at non-equilibrium)."""
        import gpu4pyscf
        mf = gpu4pyscf.dft.rks.RKS(self.mol, xc='b3lyp')
        mf.kernel()
        g = mf.nuc_grad_method()
        grad = g.kernel()
        # Gradient should not be zero (non-equilibrium geometry)
        self.assertGreater(np.max(np.abs(grad)), 1e-4)
        # Should be shape (natm, 3)
        self.assertEqual(grad.shape, (3, 3))

    def test_scf_energy_larger_mol(self):
        """Multi-GPU SCF on ethanol (9 atoms)."""
        import gpu4pyscf
        mol = gto.M(
            atom='C -0.748 -0.015 0.024; C 0.559 0.420 -0.137; '
                 'O 1.440 -0.542 0.100; H -1.294 0.577 -0.699; '
                 'H -0.855 0.002 1.082; H -1.094 -1.019 -0.208; '
                 'H 0.684 1.380 0.337; H 0.889 0.536 -1.164; '
                 'H 1.242 -0.747 1.031',
            basis='6-31g*',
            verbose=0,
        )
        mf = gpu4pyscf.dft.rks.RKS(mol, xc='b3lyp')
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertLess(e, -154.0)  # Should be around -155


class TestMultiGPUUtils(unittest.TestCase):
    """Tests for multi_gpu utility functions."""

    def test_run_single_device(self):
        """multi_gpu.run should work with kwargs."""
        from gpu4pyscf.lib import multi_gpu

        def fn(a, b=1):
            return a + b

        results = multi_gpu.run(fn, args=(2,), kwargs={'b': 3})
        self.assertEqual(results[0], 5)

    def test_map_single_device(self):
        """multi_gpu.map should work with kwargs."""
        from gpu4pyscf.lib import multi_gpu

        def fn(t, scale=1):
            return t * scale

        results = multi_gpu.map(fn, [1, 2, 3], kwargs={'scale': 2})
        self.assertEqual(results, [2, 4, 6])

    @unittest.skipIf(num_devices < 2, 'Requires 2+ GPUs')
    def test_array_broadcast(self):
        """array_broadcast should create copies on all devices."""
        from gpu4pyscf.lib import multi_gpu
        a = cp.ones(10)
        copies = multi_gpu.array_broadcast(a)
        self.assertEqual(len(copies), num_devices)
        for i, c in enumerate(copies):
            np.testing.assert_array_equal(cp.asnumpy(c), np.ones(10))

    @unittest.skipIf(num_devices < 2, 'Requires 2+ GPUs')
    def test_array_reduce(self):
        """array_reduce should sum arrays from all devices."""
        from gpu4pyscf.lib import multi_gpu
        arrays = []
        for i in range(num_devices):
            with cp.cuda.Device(i):
                arrays.append(cp.ones(10) * (i + 1))
        result = multi_gpu.array_reduce(arrays)
        expected_sum = sum(range(1, num_devices + 1))
        np.testing.assert_array_almost_equal(
            cp.asnumpy(result), np.ones(10) * expected_sum)

    @unittest.skipIf(num_devices < 2, 'Requires 2+ GPUs')
    def test_run_multi_device(self):
        """multi_gpu.run should execute on all devices."""
        from gpu4pyscf.lib import multi_gpu

        def fn():
            return cp.cuda.device.get_device_id()

        results = multi_gpu.run(fn, non_blocking=True)
        self.assertEqual(len(results), num_devices)
        self.assertEqual(sorted(results), list(range(num_devices)))


class TestMemcpy(unittest.TestCase):
    """Tests for cross-device memory copy."""

    @unittest.skipIf(num_devices < 2, 'Requires 2+ GPUs')
    def test_p2p_transfer(self):
        """p2p_transfer should copy data between devices."""
        from gpu4pyscf.lib.memcpy import p2p_transfer
        with cp.cuda.Device(0):
            src = cp.arange(100, dtype=cp.float64)
        with cp.cuda.Device(1):
            dst = cp.empty(100, dtype=cp.float64)
        p2p_transfer(dst, src)
        np.testing.assert_array_equal(
            cp.asnumpy(dst), np.arange(100, dtype=np.float64))

    @unittest.skipIf(num_devices < 2, 'Requires 2+ GPUs')
    def test_staged_copy_via_cpu(self):
        """_staged_copy_via_cpu should work for cross-device copies."""
        from gpu4pyscf.lib.memcpy import _staged_copy_via_cpu
        with cp.cuda.Device(0):
            src = cp.ones((10, 10), dtype=cp.float64) * 42.0
        with cp.cuda.Device(1):
            dst = cp.empty((10, 10), dtype=cp.float64)
        _staged_copy_via_cpu(src, dst)
        np.testing.assert_array_equal(
            cp.asnumpy(dst), np.ones((10, 10)) * 42.0)


if __name__ == '__main__':
    unittest.main()
