# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import nksr
import torch

from pycg import vis, exp
from pathlib import Path
import numpy as np
from common import load_waymo_example, warning_on_low_memory


if __name__ == '__main__':
    warning_on_low_memory(20000.0)
    xyz_np, sensor_np = load_waymo_example()

    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")

    input_xyz = torch.from_numpy(xyz_np).float().to(device)
    input_sensor = torch.from_numpy(sensor_np).float().to(device)

    field = reconstructor.reconstruct(
        input_xyz, sensor=input_sensor, detail_level=None,
        # Minor configs for better efficiency (not necessary)
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
        # Chunked reconstruction (if OOM)
        # chunk_size=51.2,
        preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0)
    )
    
    # (Optional) Convert to CPU for mesh extraction
    # field.to_("cpu")
    # reconstructor.network.to("cpu")

    mesh = field.extract_dual_mesh(mise_iter=1)
    mesh = vis.mesh(mesh.v, mesh.f)

    vis.show_3d([mesh], [vis.pointcloud(xyz_np)])
