# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from typing import Union

import torch
from pycg import exp
import numpy as np
from nksr.fields.base_field import BaseField, EvaluationResult

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    exp.logger.warning("pykdtree import error. PCNNField may not work as expected.")


class PCNNField(BaseField):
    """
    Point-cloud-based nearest-neighbour field usually for textures.
        Does not support GPU for now.
    """
    def __init__(self, pc_pos: torch.Tensor, pc_field: torch.Tensor):
        super().__init__(None)
        self.kdtree = KDTree(pc_pos.detach().cpu().numpy())
        self.pc_field = pc_field

    def to_(self, device: Union[torch.device, str]):
        super().to_(device)
        self.pc_field = self.pc_field.to(device)

    def evaluate_f(self, xyz: torch.Tensor, grad: bool = False):
        assert not grad, "PCNNField does not support gradient!"
        _, idx = self.kdtree.query(xyz.detach().cpu().numpy())
        idx = torch.from_numpy(idx.astype(np.int64)).to(self.pc_field.device)
        return EvaluationResult(self.pc_field[idx])
