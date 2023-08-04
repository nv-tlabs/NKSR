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

import point_cloud_utils as pcu


def normal_func(xyz: torch.Tensor, normal: torch.Tensor, sensor: torch.Tensor):
    assert normal is None, "normal already exists"
    assert sensor is not None, "please provide sensor positions for consistent orientations"

    xyz_numpy = xyz.cpu().numpy()
    indices, normal = pcu.estimate_point_cloud_normals_knn(xyz_numpy, 64)
    normal = torch.from_numpy(normal).to(xyz)
    indices = torch.from_numpy(indices).to(xyz).long()

    xyz, sensor = xyz[indices], sensor[indices]

    view_dir = sensor - xyz
    view_dir = view_dir / (torch.linalg.norm(view_dir, dim=-1, keepdim=True) + 1e-6)
    cos_angle = torch.sum(view_dir * normal, dim=1)
    cos_mask = cos_angle < 0.0
    normal[cos_mask] = -normal[cos_mask]

    keep_mask = torch.abs(cos_angle) > np.cos(np.deg2rad(85.0))
    xyz, normal = xyz[keep_mask], normal[keep_mask]

    return xyz, normal, None


if __name__ == '__main__':
    warning_on_low_memory(20000.0)
    xyz_np, sensor_np = load_waymo_example()

    device = torch.device("cpu")
    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")

    input_xyz = torch.from_numpy(xyz_np).float().to(device)
    input_sensor = torch.from_numpy(sensor_np).float().to(device)

    field = reconstructor.reconstruct(
        input_xyz, sensor=input_sensor, detail_level=None,
        # Minor configs for better efficiency (not necessary)
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
        # chunk_size=51.2,
        preprocess_fn=normal_func
    )

    mesh = field.extract_dual_mesh(mise_iter=1)
    mesh = vis.mesh(mesh.v, mesh.f)

    vis.show_3d([mesh], [vis.pointcloud(xyz_np)])
