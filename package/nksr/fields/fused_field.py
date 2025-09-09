# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import torch
import torch_scatter
from typing import List, Union
from pycg.isometry import Isometry
from nksr.svh import SparseFeatureHierarchy
from nksr.fields.base_field import BaseField, EvaluationResult


class FusedField(BaseField):
    def __init__(self,
                 fields: List[BaseField],
                 transforms: List[Isometry],
                 svh: SparseFeatureHierarchy = None,
                 reduce: str = "trimmer_mean"):

        # Create SVH if not provided
        if svh is None:
            assert fields is not None
            assert len(fields) == len(transforms)
            svh = SparseFeatureHierarchy.joined([f.svh for f in fields], transforms)

        super().__init__(svh)
        self.reduce = reduce
        self.fields = fields
        self.transforms = transforms

        mask_fields = [f.mask_field for f in self.fields]
        if any([m is not None for m in mask_fields]):
            self.set_mask_field(
                FusedField(mask_fields, self.transforms, self.svh, reduce="min")
            )

        self.set_level_set(self.fields[0].level_set)

        # Simple bounding box acceleration
        self.bounds = [f.svh.get_f_bound() for f in self.fields]

    def set_scale(self, scale: float):
        raise ValueError("Scale is not supported for FusedField!")

    def to_(self, device: Union[torch.device, str]):
        super().to_(device)
        for f in self.fields:
            f.to_(device)
        self.bounds = [(s.to(device), t.to(device)) for (s, t) in self.bounds]

    def evaluate_f(self, xyz: torch.Tensor, grad: bool = False):
        assert not grad, "Gradient computation not supported yet!"

        index_list = []
        f_list, mask_list = [], []

        for (field, transform, (bound_min, bound_max)) in zip(self.fields, self.transforms, self.bounds):
            local_xyz = transform.inv() @ xyz
            valid_inds = torch.logical_and(
                torch.all(local_xyz > bound_min[None, :], dim=1),
                torch.all(local_xyz < bound_max[None, :], dim=1))
            valid_inds = torch.where(valid_inds)[0]
            if valid_inds.size(0) == 0:
                continue

            index_list.append(valid_inds)

            valid_local_xyz = local_xyz[valid_inds]
            valid_f = field.evaluate_f(valid_local_xyz, grad)
            f_list.append(valid_f)

            if "trimmer" in self.reduce and field.mask_field is not None:
                mask_value = field.mask_field.evaluate_f(valid_local_xyz).value
                mask_list.append(mask_value - field.mask_field.level_set * 4.0)

        # Aggregation
        index_list = torch.cat(index_list)
        f_list = torch.cat([f.value for f in f_list])

        n_pts = xyz.size(0)
        if self.reduce == "min":
            f = torch_scatter.scatter_min(f_list, index_list, dim_size=n_pts)[0]
        elif self.reduce == "trimmer_mean":
            mask_list = torch.clamp(-torch.cat(mask_list), min=0.0)
            weight_sum = torch_scatter.scatter_sum(mask_list, index_list, dim_size=n_pts)
            weight_f = torch_scatter.scatter_sum(mask_list * f_list, index_list, dim_size=n_pts)
            f = weight_f / weight_sum
            f[torch.isnan(f)] = 0.0
        elif self.reduce == "mean":
            f = torch_scatter.scatter_mean(f_list, index_list, dim_size=n_pts)
        else:
            raise NotImplementedError

        return EvaluationResult(f)
