# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import math
import torch
import torch.nn as nn
from typing import Optional, Union
from nksr.svh import SparseFeatureHierarchy, KernelMap, VoxelStatus
from nksr import ext


class Conv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 bias: bool = True,
                 transposed: bool = False,
                 cache_kmap: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.transposed = transposed
        self.cache_kmap = cache_kmap

        self.kernel_volume = self.kernel_size ** 3
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.stride != 1:
            s += ', stride={stride}'
        if self.bias is None:
            s += ', bias=False'
        if self.transposed:
            s += ', transposed=True'
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume)
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def _compute_conv_args(self, in_grid, out_grid):
        if not self.transposed:
            kmap = ext.conv.convolution_kernel_map(in_grid._grid, out_grid._grid, self.kernel_size).T
            shape = (in_grid.num_voxels, out_grid.num_voxels)
        else:
            kmap = ext.conv.convolution_kernel_map(out_grid._grid, in_grid._grid, self.kernel_size).T
            shape = (out_grid.num_voxels, in_grid.num_voxels)

        nbsizes = torch.sum(kmap != -1, dim=1)
        nbmap = torch.nonzero(kmap != -1).contiguous()

        indices = nbmap[:, 0] * kmap.size(1) + nbmap[:, 1]
        # {in, out}
        nbmap[:, 0] = kmap.reshape(-1)[indices]
        return nbmap.int(), nbsizes.int(), shape

    def forward(self,
                in_feature: torch.Tensor,
                in_svh: SparseFeatureHierarchy,
                in_depth: int,
                out_svh: Optional[SparseFeatureHierarchy] = None,
                out_depth: Optional[int] = None):

        if out_svh is None:
            out_svh = in_svh

        if out_depth is None:
            depth_delta = round(math.log2(self.stride))
            out_depth = in_depth + depth_delta if not self.transposed else in_depth - depth_delta

        in_grid = in_svh.grids[in_depth]
        out_grid = out_svh.grids[out_depth]
        assert in_grid is not None, "grid for input branch is not built!"
        assert out_grid is not None, f"Doesn't support online branch building yet. " \
                                    f"Please call out_svh.build_xxx(...) first!"

        stride = round(out_grid.voxel_size / in_grid.voxel_size) if not self.transposed else \
            round(in_grid.voxel_size / out_grid.voxel_size)
        assert stride == self.stride, "Stride does not satisfy provided I/O branches!"

        if self.kernel_size == 1 and self.stride == 1:
            out_feature = in_feature.matmul(self.kernel)
        else:
            if in_svh is out_svh and in_depth == out_depth:
                kmap_key = (in_depth, self.kernel_size)
                if kmap_key not in in_svh.kernel_maps:
                    nbmap, nbsizes, _ = self._compute_conv_args(in_grid, out_grid)
                    if self.cache_kmap:
                        in_svh.kernel_maps[kmap_key] = KernelMap(nbmap, nbsizes)
                else:
                    kmap_obj = in_svh.kernel_maps[kmap_key]
                    nbmap, nbsizes = kmap_obj.nbmap, kmap_obj.nbsizes
                shape = (in_grid.num_voxels, in_grid.num_voxels)
            else:
                nbmap, nbsizes, shape = self._compute_conv_args(in_grid, out_grid)

            in_feature = in_feature.contiguous()
            out_feature = ext.conv.sparse_convolution(
                in_feature, self.kernel, nbmap, nbsizes, shape, self.transposed)

        if self.bias is not None:
            out_feature += self.bias

        return out_feature, out_svh, out_depth


class Activation(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self,
                in_feature: torch.Tensor,
                in_svh: SparseFeatureHierarchy,
                in_depth: int,
                out_svh: Optional[SparseFeatureHierarchy] = None,
                out_depth: Optional[int] = None):
        if out_svh is not None:
            assert in_svh is out_svh and in_depth == out_depth, "Sparse activation should be on the same layer!"
        else:
            out_svh, out_depth = in_svh, in_depth

        out_feature = self.module(in_feature)
        return out_feature, out_svh, out_depth


class GroupNorm(nn.GroupNorm):
    def forward(self,
                feat: torch.Tensor,
                in_svh: SparseFeatureHierarchy,
                in_depth: int,
                out_svh: Optional[SparseFeatureHierarchy] = None,
                out_depth: Optional[int] = None):
        if out_svh is not None:
            assert in_svh is out_svh and in_depth == out_depth, "Sparse GroupNorm should be on the same layer!"
        else:
            out_svh, out_depth = in_svh, in_depth

        num_channels = feat.size(1)
        feat = feat.transpose(0, 1).reshape(1, num_channels, -1)
        feat = super().forward(feat)
        feat = feat.reshape(num_channels, -1).transpose(0, 1)
        return feat, out_svh, out_depth


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, latent_dim: int):
        super().__init__()
        self.group_norm = GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.linear_factor = nn.Linear(latent_dim, num_channels)
        self.linear_bias = nn.Linear(latent_dim, num_channels)
        torch.nn.init.kaiming_uniform_(self.linear_factor.weight, a=0.1)
        self.linear_factor.bias.data[:] = 1.0
        torch.nn.init.kaiming_uniform_(self.linear_bias.weight, a=0.1)
        self.linear_bias.bias.data[:] = 0.0

    def extra_repr(self) -> str:
        return self.group_norm.extra_repr() + ", with Factor(" + self.linear_factor.extra_repr() + ")"

    def forward(self,
                feat: torch.Tensor,
                in_svh: SparseFeatureHierarchy,
                in_depth: int,
                out_svh: Optional[SparseFeatureHierarchy] = None,
                out_depth: Optional[int] = None,
                latent: Optional[torch.Tensor] = None):
        feat, out_svh, out_depth = self.group_norm(feat, in_svh, in_depth, out_svh, out_depth)
        factor, bias = self.linear_factor(latent), self.linear_bias(latent)
        feat = feat * factor + bias
        return feat, out_svh, out_depth


class Upsampling(nn.Module):
    def __init__(self, scale_factor: int, mode: str = "nearest"):
        super().__init__()
        assert mode in ['nearest', 'trilinear']
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self,
                feat: torch.Tensor,
                in_svh: SparseFeatureHierarchy,
                in_depth: int,
                out_svh: Optional[SparseFeatureHierarchy] = None,
                out_depth: Optional[int] = None,
                mask: Optional[torch.Tensor] = None):

        if out_svh is None:
            out_svh = in_svh

        if out_depth is None:
            out_depth = in_depth - round(math.log2(self.scale_factor))
            assert out_depth >= 0, "Cannot infer out_depth for the upsample operation!"

        in_grid = in_svh.grids[in_depth]
        out_grid = out_svh.grids[out_depth]
        if out_grid is None:

            if mask is None:
                mask = torch.ones(in_grid.num_voxels, dtype=bool, device=out_svh.device)

            out_grid = in_grid.subdivided_grid(self.scale_factor, mask)
            out_svh.build_from_grid(out_depth, out_grid)

        else:
            assert round(in_grid.voxel_size / out_grid.voxel_size) == self.scale_factor
            assert mask is None, "Mask cannot be guaranteed when target hierarchy exists!"

        if self.mode == 'nearest':
            feat, _ = in_grid.subdivide(feat, self.scale_factor, fine_grid=out_grid)
        elif self.mode == 'trilinear':
            # Note: This is different from Dense tensor's version in that:
            #   1. align_corners = False
            #   2. Used zero padding instead of boundary padding.
            out_pos = out_grid.grid_to_world(out_grid.active_grid_coords().float())
            feat = in_grid.sample_trilinear(out_pos, feat)
        else:
            raise NotImplementedError

        return feat, out_svh, out_depth


class MaxPooling(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self,
                feat: torch.Tensor,
                in_svh: SparseFeatureHierarchy,
                in_depth: int,
                out_svh: Optional[SparseFeatureHierarchy] = None,
                out_depth: Optional[int] = None):

        if out_svh is None:
            out_svh = in_svh

        if out_depth is None:
            out_depth = in_depth + round(math.log2(self.kernel_size))

        in_grid = in_svh.grids[in_depth]
        out_grid = out_svh.grids[out_depth]
        if out_grid is None:
            out_svh.build_from_grid(out_depth, in_grid.coarsened_grid(self.kernel_size))
            out_grid = out_svh.grids[out_depth]
        else:
            assert round(out_grid.voxel_size / in_grid.voxel_size) == self.kernel_size

        feat, _ = in_grid.max_pool(feat, self.kernel_size, coarse_grid=out_grid)
        feat[torch.isinf(feat)] = 0.0

        return feat, out_svh, out_depth


class SparseZeroPadding(nn.Module):
    def forward(self,
                feat: torch.Tensor,
                in_svh: SparseFeatureHierarchy,
                in_depth: int,
                out_svh: Optional[SparseFeatureHierarchy],
                out_depth: Optional[int] = None):

        if out_depth is None:
            out_depth = in_depth

        assert in_depth == out_depth, "Padding output should be at the same depth!"

        in_grid, out_grid = in_svh.grids[in_depth], out_svh.grids[out_depth]

        if in_grid is out_grid:
            out_feat = feat
        else:
            if feat.ndim == 1:
                out_feat = torch.zeros(out_grid.num_voxels, device=feat.device, dtype=feat.dtype)
            else:
                out_feat = torch.zeros((out_grid.num_voxels, feat.size(1)), device=feat.device, dtype=feat.dtype)
            in_idx = in_grid.ijk_to_index(out_grid.active_grid_coords())
            in_mask = in_idx != -1
            out_feat[in_mask] = feat[in_idx[in_mask]]

        return out_feat, out_svh, out_depth


class SparseSequential(nn.Module):
    def forward(self,
                feat: torch.Tensor,
                svh: SparseFeatureHierarchy,
                depth: int,
                **kwargs):
        for module in self._modules.values():
            feat, svh, depth = module(feat, svh, depth, **kwargs)
        return feat, svh, depth
