# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch_scatter
from nksr.ext import meshing


class MarchingCubes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cube_corner_inds, corner_pos, corner_value):
        # v: (V, 3) float | f: (T, 3) long | vidx: (V, 2) long
        assert torch.max(cube_corner_inds) < corner_pos.size(0) == corner_value.size(0)
        v, f, vidx = meshing.marching_cubes(cube_corner_inds, corner_pos, corner_value)
        ctx.save_for_backward(corner_pos, corner_value, vidx)
        return v, f

    @staticmethod
    def backward(ctx, grad_v, grad_f):
        corner_pos, corner_value, vidx = ctx.saved_tensors
        s1, s2 = corner_value[vidx[:, 0]], corner_value[vidx[:, 1]]
        v1, v2 = corner_pos[vidx[:, 0]], corner_pos[vidx[:, 1]]
        d_common = torch.sum((v1 - v2) * grad_v, dim=1) / (s2 - s1) ** 2
        dv_ds1 = s2 * d_common         # (V,)
        dv_ds2 = -s1 * d_common        # (V,)
        grad_s1 = torch_scatter.scatter_sum(dv_ds1, vidx[:, 0], dim_size=corner_value.size(0))
        grad_s2 = torch_scatter.scatter_sum(dv_ds2, vidx[:, 1], dim_size=corner_value.size(0))
        return None, None, grad_s1 + grad_s2
