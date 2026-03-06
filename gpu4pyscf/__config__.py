# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import os
import logging
import cupy

logger = logging.getLogger(__name__)

num_devices = cupy.cuda.runtime.getDeviceCount()

props = cupy.cuda.runtime.getDeviceProperties(0)
GB = 1024*1024*1024
min_ao_blksize = 256        # maxisum batch size of AOs
min_grid_blksize = 64*64    # maximum batch size of grids for DFT
ao_aligned = 32             # global AO alignment for slicing
grid_aligned = 256          # 256 alignment for grids globally

# Use smaller blksize for old gaming GPUs
if props['totalGlobalMem'] < 16 * GB:
    min_ao_blksize = 64
    min_grid_blksize = 64*64

# Use 90% of the global memory for CuPy memory pool
mem_fraction = 0.9
cupy.get_default_memory_pool().set_limit(fraction=mem_fraction)

if props['sharedMemPerBlockOptin'] > 65536:
    shm_size = props['sharedMemPerBlockOptin']
else:
    shm_size = props['sharedMemPerBlock']

# Check P2P data transfer is available
_p2p_access = True
_p2p_enabled = {}  # Track which device pairs have p2p enabled
if num_devices > 1:
    for src in range(num_devices):
        for dst in range(num_devices):
            if src != dst:
                try:
                    can_access_peer = cupy.cuda.runtime.deviceCanAccessPeer(src, dst)
                    _p2p_access &= bool(can_access_peer)
                    if can_access_peer:
                        # Enable peer access between devices
                        try:
                            with cupy.cuda.Device(src):
                                cupy.cuda.runtime.deviceEnablePeerAccess(dst)
                            _p2p_enabled[(src, dst)] = True
                        except cupy.cuda.runtime.CUDARuntimeError as e:
                            if 'PeerAccessAlreadyEnabled' in str(e):
                                _p2p_enabled[(src, dst)] = True
                            else:
                                logger.warning(
                                    'Failed to enable P2P access %d->%d: %s', src, dst, e)
                                _p2p_access = False
                                _p2p_enabled[(src, dst)] = False
                except Exception as e:
                    logger.warning('P2P access check failed for %d->%d: %s', src, dst, e)
                    _p2p_access = False

    if _p2p_access:
        logger.info('P2P access enabled across all %d devices', num_devices)
    else:
        logger.info('P2P access not fully available across %d devices, '
                     'using staged CPU transfers for cross-device copies', num_devices)

# Allow override via environment variable
_force_single_gpu = os.environ.get('GPU4PYSCF_SINGLE_GPU', '').lower() in ('1', 'true', 'yes')
if _force_single_gpu and num_devices > 1:
    logger.info('GPU4PYSCF_SINGLE_GPU set, forcing single-GPU mode (device 0)')
    num_devices = 1

# Overwrite the above settings using the global pyscf configs
from pyscf.__config__ import * # noqa
