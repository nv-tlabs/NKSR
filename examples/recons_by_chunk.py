# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np
import nksr
from pycg import vis
from common import load_buda_example, warning_on_low_memory


if __name__ == '__main__':
    # chunking reduces 1G memory footprint, if you reduce chunk_size to 20.0, then only 3.5G memory is needed.
    warning_on_low_memory(1024.0 * 7.0)
    device = torch.device("cuda:0")

    buda_geom = load_buda_example()

    input_xyz = torch.from_numpy(np.asarray(buda_geom.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(buda_geom.normals)).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")

    field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=None, chunk_size=50.0)
    mesh = field.extract_dual_mesh(mise_iter=1)

    vis.show_3d([vis.mesh(mesh.v, mesh.f)], [buda_geom])
