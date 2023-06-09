# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np
from pycg import vis, exp
from nksr import Reconstructor, utils, fields
from common import load_spot_example, warning_on_low_memory


if __name__ == '__main__':
    device = torch.device("cuda:0")

    test_geom = load_spot_example()

    input_xyz = torch.from_numpy(np.asarray(test_geom.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(test_geom.normals)).float().to(device)
    input_color = torch.from_numpy(np.asarray(test_geom.colors)).float().to(device)

    nksr = Reconstructor(device)

    field = nksr.reconstruct(input_xyz, input_normal, detail_level=1.0)
    field.set_texture_field(fields.PCNNField(input_xyz, input_color))

    mesh = field.extract_dual_mesh(max_points=2 ** 22, mise_iter=1)
    mesh = vis.mesh(mesh.v, mesh.f, color=mesh.c)

    vis.show_3d([mesh], [test_geom])
