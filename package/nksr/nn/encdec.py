# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


# Code adapted from Convolutional Occupancy Networks.
#   As some arguments are found to be less improving, we removed them.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from nksr.svh import SparseFeatureHierarchy


class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class. """
    def __init__(self, size_in: int, size_out: int = None, size_h: int = None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class PointEncoder(nn.Module):
    """ PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    """

    def __init__(self,
                 dim: int,
                 c_dim: int = 32,
                 hidden_dim: int = 32,
                 n_blocks: int = 3):
        super().__init__()

        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim)
            for _ in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.hidden_dim = hidden_dim

    def forward(self,
                pts_xyz: torch.Tensor,
                pts_feature: torch.Tensor,
                svh: SparseFeatureHierarchy,
                depth: int = 0):

        grid = svh.grids[depth]
        assert grid is not None, "Grid structure is not built for PointEncoder!"

        # Get voxel idx
        pts_xyz = grid.world_to_grid(pts_xyz)
        vid = grid.ijk_to_index(pts_xyz.round().int())

        # Map coordinates to local voxel
        pts_xyz = (pts_xyz + 0.5) % 1
        pts_mask = vid != -1
        vid, pts_xyz = vid[pts_mask], pts_xyz[pts_mask]

        # Feature extraction
        if pts_feature is None:
            pts_feature = self.fc_pos(pts_xyz)
        else:
            pts_feature = pts_feature[pts_mask]
            pts_feature = self.fc_pos(torch.cat([pts_xyz, pts_feature], dim=1))
        pts_feature = self.blocks[0](pts_feature)
        for block in self.blocks[1:]:
            pooled = scatter_max(pts_feature, vid, dim=0, dim_size=grid.num_voxels)[0]
            pooled = pooled[vid]
            pts_feature = torch.cat([pts_feature, pooled], dim=1)
            pts_feature = block(pts_feature)

        c = self.fc_c(pts_feature)
        c = scatter_mean(c, vid, dim=0, dim_size=grid.num_voxels)
        return c


class MultiscalePointDecoder(nn.Module):
    def __init__(self,
                 c_each_dim: int = 16,
                 multiscale_depths: int = 4,
                 p_dim: int = 3,
                 out_dim: int = 1,
                 hidden_size: int = 32,
                 n_blocks: int = 2,
                 aggregation: str = 'cat',
                 out_init: float = None,
                 coords_depths: list = None):

        if aggregation == 'cat':
            c_dim = c_each_dim * multiscale_depths
        elif aggregation == 'sum':
            c_dim = c_each_dim
        else:
            raise NotImplementedError

        if coords_depths is None:
            coords_depths = list(range(multiscale_depths))
        coords_depths = sorted(coords_depths)

        super().__init__()
        self.c_dim = c_dim
        self.c_each_dim = c_each_dim
        self.n_blocks = n_blocks
        self.multiscale_depths = multiscale_depths
        self.aggregation = aggregation
        self.coords_depths = coords_depths

        self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)])
        self.fc_p = nn.Linear(p_dim * len(coords_depths), hidden_size)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for _ in range(n_blocks)
        ])
        self.fc_out = nn.Linear(hidden_size, out_dim)
        self.out_dim = out_dim

        # Init parameters
        if out_init is not None:
            nn.init.zeros_(self.fc_out.weight)
            nn.init.constant_(self.fc_out.bias, out_init)

    def forward(self,
                xyz: torch.Tensor,
                svh: SparseFeatureHierarchy,
                multiscale_feat: dict):

        p_feats = []
        for did in self.coords_depths:
            vs = svh.grids[did].voxel_size
            p = (xyz % vs) / vs - 0.5
            p_feats.append(p)
        p = torch.cat(p_feats, dim=1)

        c_feats = []
        for did in range(self.multiscale_depths):
            if svh.grids[did] is None:
                c = torch.zeros((xyz.size(0), self.c_each_dim), device=xyz.device)
            else:
                c = svh.grids[did].sample_trilinear(xyz, multiscale_feat[did])
            c_feats.append(c)

        if self.aggregation == 'cat':
            c = torch.cat(c_feats, dim=1)
        else:
            c = sum(c_feats)

        net = self.fc_p(p)
        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(F.relu(net))

        return out
