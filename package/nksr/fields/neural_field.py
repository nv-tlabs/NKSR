# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from typing import Union

import torch
from nksr.nn.encdec import MultiscalePointDecoder
from nksr.svh import SparseFeatureHierarchy
from nksr.fields.base_field import BaseField, EvaluationResult


class NeuralField(BaseField):
    def __init__(self,
                 svh: SparseFeatureHierarchy,
                 decoder: MultiscalePointDecoder,
                 features: dict,
                 grad_type: str = "numerical"):
        super().__init__(svh)
        self.decoder = decoder
        self.features = features

        assert grad_type in ["numerical", "analytical"]
        self.grad_type = grad_type

    def to_(self, device: Union[torch.device, str]):
        super().to_(device)
        self.features = {k: v.to(device) for k, v in self.features.items()}

    def evaluate_f(self, xyz: torch.Tensor, grad: bool = False):
        res = self.decoder(xyz, self.svh, self.features)
        assert res.size(1) == 1, "Decoder is only allowed to produce 1-dim vectors."

        if grad:
            if self.grad_type == "numerical":
                interval = 0.01 * self.svh.voxel_size
                grad_value = []
                for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
                    offset_tensor = torch.tensor(offset, device=self.device)[None, :]
                    res_p = self.decoder(xyz + offset_tensor, self.svh, self.features)[:, 0]
                    res_n = self.decoder(xyz - offset_tensor, self.svh, self.features)[:, 0]
                    grad_value.append((res_p - res_n) / (2 * interval))
                grad_value = torch.stack(grad_value, dim=1)
            else:
                xyz_d = torch.clone(xyz)
                xyz_d.requires_grad = True
                with torch.enable_grad():
                    res_d = self.decoder(xyz_d, self.svh, self.features)
                    grad_value = torch.autograd.grad(res_d, [xyz_d],
                                                     grad_outputs=torch.ones_like(res_d),
                                                     create_graph=self.decoder.training)[0]
        else:
            grad_value = None

        return EvaluationResult(res[:, 0], grad_value)
