# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import torch
from nksr.svh import SparseFeatureHierarchy
from nksr.fields.base_field import BaseField, EvaluationResult


class LayerField(BaseField):
    """
    Provides 1 for regions outside layer's voxel boundary, -1 for regions inside (including the exact boundary)
    """
    def __init__(self,
                 svh: SparseFeatureHierarchy,
                 inside_depth: int):
        super().__init__(svh)
        self.inside_depth = inside_depth

    def evaluate_f(self, xyz: torch.Tensor, grad: bool = False):
        assert not grad, "Layer step function does not have valid gradient"
        in_grid_mask = self.svh.grids[self.inside_depth - 1].points_in_active_voxel(xyz)
        f = torch.ones(xyz.size(0), device=xyz.device)
        f[in_grid_mask] = -1
        return EvaluationResult(f)
