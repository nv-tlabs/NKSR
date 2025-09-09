# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import torch
from abc import ABC
import numpy as np
from typing import Union, Optional
from nksr.svh import SparseFeatureHierarchy
from nksr.meshing import MarchingCubes
from nksr.ext import meshing
from nksr import utils


class EvaluationResult:
    def __init__(self, value: torch.Tensor, gradient: torch.Tensor = None):
        self.value = value
        self.gradient = gradient

    @classmethod
    def zero(cls, grad: bool = False):
        return EvaluationResult(0, 0 if grad else None)

    def __add__(self, other):
        assert isinstance(other, EvaluationResult)
        return EvaluationResult(
            self.value + other.value,
            (self.gradient + other.gradient) if self.gradient is not None else None
        )

    def __sub__(self, other):
        assert not isinstance(other, EvaluationResult)
        return EvaluationResult(
            self.value - other,
            self.gradient
        )


class MeshingResult:
    def __init__(self, v: torch.Tensor = None, f: torch.Tensor = None, c: torch.Tensor = None):
        self.v = v
        self.f = f
        self.c = c


class BaseField(ABC):
    """
    Base class for the 3D continuous field:
        f_bar = f - level_set
    """
    def __init__(self, svh: Optional[SparseFeatureHierarchy]):
        self.svh = svh
        self.scale = 1.0
        self.mask_field = None
        self.texture_field = None
        self.set_level_set(0.0)

    def set_level_set(self, level_set: float):
        self.level_set = level_set

    def set_mask_field(self, mask_field: "BaseField"):
        self.mask_field = mask_field

    def set_texture_field(self, texture_field: "BaseField"):
        self.texture_field = texture_field

    def set_scale(self, scale: float):
        self.scale = scale
        if self.mask_field is not None:
            self.mask_field.set_scale(scale)
        if self.texture_field is not None:
            self.texture_field.set_scale(scale)

    def clear_svh_kernel_maps(self):
        if self.svh is not None:
            self.svh.clear_kernel_maps()
        if self.mask_field is not None:
            self.mask_field.clear_svh_kernel_maps()
        if self.texture_field is not None:
            self.texture_field.clear_svh_kernel_maps()

    def to_(self, device: Union[torch.device, str]):
        if self.svh is not None:
            self.svh.to_(device)
        if self.mask_field is not None:
            self.mask_field.to_(device)
        if self.texture_field is not None:
            self.texture_field.to_(device)

    @property
    def device(self):
        return self.svh.device

    def evaluate_f(self, xyz: torch.Tensor, grad: bool = False):
        pass

    def evaluate_f_bar(self, xyz: torch.Tensor, max_points: int = -1, verbose: bool = True):
        n_chunks = int(np.ceil(xyz.size(0) / max_points)) if max_points > 0 else 1
        xyz_chunks = torch.chunk(xyz, n_chunks)
        f_bar_chunks = []

        if verbose and len(xyz_chunks) > 10:
            from tqdm import tqdm
            xyz_chunks = tqdm(xyz_chunks)

        for xyz_chunk in xyz_chunks:
            if self.scale != 1.0:
                xyz_chunk = xyz_chunk / self.scale
            f_chunk = self.evaluate_f(xyz_chunk, grad=False).value
            f_bar_chunks.append(f_chunk - self.level_set)

        return torch.cat(f_bar_chunks)

    def extract_primal_mesh(self, depth: int, resolution: int = 2, trim: bool = True, max_points: int = -1):
        primal_grid = self.svh.grids[depth]
        primal_grid_dense = primal_grid.subdivided_grid(
            resolution,
            torch.ones(primal_grid.num_voxels, dtype=bool, device=self.svh.device))
        dual_grid_dense = primal_grid_dense.dual_grid()

        dual_graph = meshing.primal_cube_graph(primal_grid_dense, dual_grid_dense)
        dual_corner_pos = dual_grid_dense.grid_to_world(dual_grid_dense.active_grid_coords().float())
        if self.scale != 1.0:
            dual_corner_pos = dual_corner_pos * self.scale
        dual_corner_value = self.evaluate_f_bar(dual_corner_pos, max_points=max_points)

        primal_v, primal_f = MarchingCubes().apply(dual_graph, dual_corner_pos, dual_corner_value)

        if self.mask_field is not None and trim:
            vert_mask = self.mask_field.evaluate_f_bar(primal_v, max_points=max_points) < 0.0
            primal_v, primal_f = utils.apply_vertex_mask(primal_v, primal_f, vert_mask)

        if self.texture_field is not None:
            primal_c = self.texture_field.evaluate_f_bar(primal_v, max_points=max_points)
        else:
            primal_c = None

        return MeshingResult(primal_v, primal_f, primal_c)

    def extract_dual_mesh(self, mise_iter: int = 0, grid_upsample: int = 1,
                          max_depth: int = 100, trim: bool = True, max_points: int = -1):
        """

        Args:
            mise_iter (int): iteration for the Multi-IsoSurface Extraction algorithm
            grid_upsample (int): number of upsample before MISE.
                final grid resolution = actual_voxel_res * grid_upsample * (2 ** mise_iter)
            max_depth:
            trim:
            max_points:

        Returns:

        """
        flattened_grids = []
        for d in range(min(self.svh.depth, max_depth + 1)):
            f_grid = meshing.build_flattened_grid(
                self.svh.grids[d]._grid,
                self.svh.grids[d - 1]._grid if d > 0 else None,
                d != self.svh.depth - 1
            )
            if grid_upsample > 1:
                f_grid = f_grid.subdivided_grid(grid_upsample)
            flattened_grids.append(f_grid)
        dual_grid = meshing.build_joint_dual_grid(flattened_grids)
        dmc_graph = meshing.dual_cube_graph(flattened_grids, dual_grid)
        dmc_vertices = torch.cat([
            f_grid.grid_to_world(f_grid.active_grid_coords().float())
            for f_grid in flattened_grids if f_grid.num_voxels() > 0
        ], dim=0)
        del flattened_grids, dual_grid

        if self.scale != 1.0:
            dmc_vertices = dmc_vertices * self.scale

        dmc_value = self.evaluate_f_bar(dmc_vertices, max_points=max_points)

        for _ in range(mise_iter):
            cube_sign = dmc_value[dmc_graph] > 0
            cube_mask = ~torch.logical_or(torch.all(cube_sign, dim=1), torch.all(~cube_sign, dim=1))
            dmc_graph = dmc_graph[cube_mask]
            unq, dmc_graph = torch.unique(dmc_graph.view(-1), return_inverse=True)
            dmc_graph = dmc_graph.view(-1, 8)
            dmc_vertices = dmc_vertices[unq]
            dmc_graph, dmc_vertices = utils.subdivide_cube_indices(dmc_graph, dmc_vertices)
            dmc_value = self.evaluate_f_bar(dmc_vertices, max_points=max_points)

        dual_v, dual_f = MarchingCubes().apply(dmc_graph, dmc_vertices, dmc_value)

        if self.mask_field is not None and trim:
            vert_mask = self.mask_field.evaluate_f_bar(dual_v, max_points=max_points) < 0.0
            dual_v, dual_f = utils.apply_vertex_mask(dual_v, dual_f, vert_mask)

        if self.texture_field is not None:
            dual_c = self.texture_field.evaluate_f_bar(dual_v, max_points=max_points)
        else:
            dual_c = None

        return MeshingResult(dual_v, dual_f, dual_c)
