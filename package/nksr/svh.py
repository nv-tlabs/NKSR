# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import torch
import numpy as np
import torch_scatter
from typing import Dict, Tuple, List, Union, Optional
from nksr.ext import _CpuIndexGrid, _CudaIndexGrid
from nksr.utils import get_device, Device
from enum import Enum
from pycg import vis
from pycg.isometry import Isometry


class KernelMap:
    def __init__(self, nbmap, nbsizes):
        self.nbmap = nbmap
        self.nbsizes = nbsizes

    def to(self, device: Device):
        device = torch.device(device)
        return KernelMap(
            self.nbmap.to(device),
            self.nbsizes.to(device)
        )


class VoxelStatus(Enum):
    # Voxel Status: 0-NE, 1-E&-, 2-E&v
    VS_NON_EXIST = 0
    VS_EXIST_STOP = 1
    VS_EXIST_CONTINUE = 2


class SparseIndexGrid:

    @classmethod
    def from_ijk_coords(cls, ijk: torch.tensor, vox_size: float, vox_origin: float,
                        pad_min: Union[list, tuple, torch.tensor, np.array] = (0, 0, 0),
                        pad_max: Union[list, tuple, torch.tensor, np.array] = (0, 0, 0),
                        device: Device = 'cpu') -> "SparseIndexGrid":
        ret = cls(vox_size, vox_origin, device)
        ret.build_from_ijk_coords(ijk, pad_min, pad_max)
        return ret

    def __init__(self, voxel_size: float, voxel_origin: float,
                 device: Device = torch.device('cpu')):
        self._device = get_device(device)
        device_idx = self._device.index if self._device.type == 'cuda' else -1
        self._cpu_grid = _CpuIndexGrid(voxel_size, voxel_origin, device_idx)
        self._cuda_grid = _CudaIndexGrid(voxel_size, voxel_origin, device_idx)

        if self._device.type == 'cuda':
            self._grid = self._cuda_grid
        elif self._device.type == 'cpu':
            self._grid = self._cpu_grid
        else:
            raise ValueError(f"Bad device type {self._device.type}")

    def __repr__(self):
        return f"SparseIndexGrid on {self.device} " \
               f"({self.num_voxels} voxels, voxel size = {self.voxel_size}, origin = {self.origin})"

    def _internal_wrap_grid(self, grid):
        ret = SparseIndexGrid(self.voxel_size, self.origin, self.device)
        if self._device.type == 'cuda':
            assert isinstance(grid, _CudaIndexGrid)
            ret._cuda_grid = grid
            ret._grid = ret._cuda_grid
        elif self._device.type == 'cpu':
            assert isinstance(grid, _CpuIndexGrid)
            ret._cpu_grid = grid
            ret._grid = ret._cpu_grid
        else:
            raise ValueError(f"Bad device type {self._device.type}")

        return ret

    def build_from_pointcloud(self, points: torch.tensor,
                              pad_min: Union[list, tuple, torch.tensor, np.array] = (0, 0, 0),
                              pad_max: Union[list, tuple, torch.tensor, np.array] = (0, 0, 0)) -> None:
        self._grid.build_from_pointcloud(points, pad_min, pad_max)

    def build_from_pointcloud_nearest_voxels(self, points: torch.tensor) -> None:
        self._grid.build_from_pointcloud_nearest_voxels(points)

    def build_from_ijk_coords(self, points: torch.tensor,
                              pad_min: Union[list, tuple, torch.tensor, np.array] = (0, 0, 0),
                              pad_max: Union[list, tuple, torch.tensor, np.array] = (0, 0, 0)) -> None:
        self._grid.build_from_ijk_coords(points, pad_min, pad_max)

    @property
    def num_voxels(self) -> int:
        return self._grid.num_voxels()

    @property
    def origin(self) -> float:
        return self._grid.origin()

    @origin.setter
    def origin(self, origin):
        self._grid.set_origin(origin)

    @property
    def voxel_size(self) -> float:
        return self._grid.voxel_size()

    @voxel_size.setter
    def voxel_size(self, voxel_size):
        self._grid.set_voxel_size(voxel_size)

    @property
    def device(self) -> torch.device:
        return self._device

    def active_grid_coords(self) -> torch.tensor:
        return self._grid.active_grid_coords()

    def points_in_active_voxel(self, points: torch.tensor) -> torch.tensor:
        return self._grid.points_in_active_voxel(points)

    def ijk_to_index(self, ijk: torch.tensor) -> torch.tensor:
        return self._grid.ijk_to_index(ijk)

    def splat_trilinear(self, points: torch.tensor, points_data: torch.tensor,
                        return_counts: bool = False) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        return self._grid.splat_trilinear(points, points_data, return_counts)

    def sample_trilinear(self, points: torch.tensor, grid_data: torch.tensor,
                         return_grad: bool = False) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        return self._grid.sample_trilinear(points, grid_data, return_grad)

    def sample_bezier(self, points: torch.tensor, grid_data: torch.tensor,
                      return_grad: bool = False) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:

        return self._grid.sample_bezier(points, grid_data, return_grad)

    def grid_to_world(self, ijk: torch.tensor) -> torch.tensor:
        return self._grid.grid_to_world(ijk)

    def world_to_grid(self, pts: torch.tensor) -> torch.tensor:
        return self._grid.world_to_grid(pts)

    def coarsened_grid(self, coarsening_factor: int) -> "SparseIndexGrid":
        if coarsening_factor <= 0:
            raise ValueError("coarsening_factor must be greater than 0")
        return self._internal_wrap_grid(self._grid.coarsened_grid(coarsening_factor))

    def subdivided_grid(self, subdivision_factor: int,
                        mask: Optional[torch.tensor] = None) -> "SparseIndexGrid":
        if subdivision_factor <= 0:
            raise ValueError("subdivision_factor must be greater than 0")
        return self._internal_wrap_grid(self._grid.subdivided_grid(subdivision_factor, mask))

    def dual_grid(self) -> "SparseIndexGrid":
        return self._internal_wrap_grid(self._grid.dual_grid())

    def subdivide(self, grid_data: torch.tensor, subdivision_factor: int,
                  fine_grid: Optional["SparseIndexGrid"] = None,
                  mask: Optional[torch.tensor] = None) -> Tuple[torch.tensor, "SparseIndexGrid"]:
        if subdivision_factor <= 0:
            raise ValueError("subdivision_factor must be greater than 0")
        if fine_grid is None:
            fine_grid = self.subdivided_grid(subdivision_factor, mask)

        return self._grid.subdivide(fine_grid._grid, subdivision_factor, grid_data), fine_grid

    def max_pool(self, grid_data: torch.tensor, pool_factor: int,
                 coarse_grid: Optional["SparseIndexGrid"] = None) -> Tuple[torch.tensor, "SparseIndexGrid"]:
        if pool_factor <= 0:
            raise ValueError("pool_factor must be greater than 0")
        if coarse_grid is None:
            coarse_grid = self.coarsened_grid(pool_factor)

        return self._grid.max_pool(coarse_grid._grid, pool_factor, grid_data), coarse_grid

    def to(self, device: Device):
        device = torch.device(device)

        if device == self.device:
            return self

        if self.device.type == device.type:
            assert(self.device.type == 'cuda')
            self._cuda_grid = self._cuda_grid.to_cuda(self.device.index)
            self._grid = self._cuda_grid
        elif device.type == 'cuda':
            self._cuda_grid = self._cpu_grid.to_cuda(device.index)
            self._grid = self._cuda_grid
        elif device.type == 'cpu':
            self._cpu_grid = self._cuda_grid.to_cpu()
            self._grid = self._cpu_grid
        else:
            raise ValueError(f"Bad device type {self._device.type}")

        self._device = device
        return self

    def __reduce__(self):
        return (self.__class__.from_ijk_coords,
                (self.active_grid_coords(), self.voxel_size, self.origin,
                 (0, 0, 0), (0, 0, 0), self.device))


class SparseFeatureHierarchy:
    """ A hierarchy of sparse grids, where voxel corners align with the origin """
    def __init__(self, voxel_size: float, depth: int, device):
        self.device = device
        self.voxel_size = voxel_size
        self.depth = depth
        self.grids: List[Optional[SparseIndexGrid]] = [None for _ in range(self.depth)]
        # ((depth, kernel_size) -> KernelMap)
        self.kernel_maps: Dict[Tuple[int, int], KernelMap] = {}

    def __repr__(self):
        text = f"SparseFeatureHierarchy - {self.depth} layers, Voxel size = {self.voxel_size}"
        if len(self.kernel_maps) > 0:
            text += f", Cached kernel maps = {sorted(list(self.kernel_maps.keys()))}"
        text += '\n'
        for d, d_grid in enumerate(self.grids):
            if d_grid is None:
                text += f"\t[{d} empty]"
            else:
                text += f"\t[{d} {d_grid.num_voxels} voxels]"
        return text + "\n"

    def tensor_dict(self):
        return {
            "voxel_size": self.voxel_size, "depth": self.depth, "device": self.device,
            "grid_coords": [
                self.grids[d].active_grid_coords() if self.grids[d] is not None else None
                for d in range(self.depth)
            ]
        }

    @classmethod
    def load_tensor_dict(cls, tensor_dict: dict):
        inst = cls(tensor_dict["voxel_size"], tensor_dict["depth"], tensor_dict["device"])
        for d in range(inst.depth):
            ijk = tensor_dict["grid_coords"][d]
            if ijk is None:
                continue
            inst.build_from_grid_coords(d, ijk)
        return inst

    def get_grid_voxel_size_origin(self, depth: int):
        return self.voxel_size * (2 ** depth), 0.5 * self.voxel_size * (2 ** depth)

    def get_voxel_centers(self, depth: int):
        grid = self.grids[depth]
        if grid is None:
            return torch.zeros((0, 3), device=self.device)
        return grid.grid_to_world(grid.active_grid_coords().float())

    def get_f_bound(self):
        grid = self.grids[self.depth - 1]
        grid_coords = grid.active_grid_coords()
        min_extent = grid.grid_to_world(torch.min(grid_coords, dim=0).values.unsqueeze(0) - 1.5)[0]
        max_extent = grid.grid_to_world(torch.max(grid_coords, dim=0).values.unsqueeze(0) + 1.5)[0]
        return min_extent, max_extent

    def evaluate_voxel_status(self, grid, depth: int):
        """
        Evaluate the voxel status of given coordinates
        :param grid: Featuregrid Grid
        :param depth: int
        :return: (N, ) byte tensor, with value 0,1,2
        """
        status = torch.full((grid.num_voxels,), VoxelStatus.VS_NON_EXIST.value,
                            dtype=torch.uint8, device=self.device)

        if self.grids[depth] is not None:
            exist_idx = grid.ijk_to_index(self.grids[depth].active_grid_coords())
            status[exist_idx[exist_idx != -1]] = VoxelStatus.VS_EXIST_STOP.value

            if depth > 0 and self.grids[depth - 1] is not None:
                child_coords = torch.div(self.grids[depth - 1].active_grid_coords(), 2, rounding_mode='floor')
                child_idx = grid.ijk_to_index(child_coords)
                status[child_idx[child_idx != -1]] = VoxelStatus.VS_EXIST_CONTINUE.value

        return status

    def get_test_grid(self, depth: int = 0, resolution: int = 2):
        grid = self.grids[depth]
        assert grid is not None
        primal_coords = grid.active_grid_coords()
        box_coords = torch.linspace(-0.5, 0.5, resolution, device=self.device)
        box_coords = torch.stack(torch.meshgrid(box_coords, box_coords, box_coords, indexing='ij'), dim=3)
        box_coords = box_coords.view(-1, 3)
        query_pos = primal_coords.unsqueeze(1) + box_coords.unsqueeze(0)
        query_pos = grid.grid_to_world(query_pos.view(-1, 3))
        return query_pos, primal_coords

    def get_visualization(self):
        wire_blocks = []
        for d in range(self.depth):
            if self.grids[d] is None:
                continue
            primal_coords = self.grids[d].active_grid_coords()
            is_lowest = len(wire_blocks) == 0
            wire_blocks.append(vis.wireframe_bbox(
                self.grids[d].grid_to_world(primal_coords - (0.45 if is_lowest else 0.5)),
                self.grids[d].grid_to_world(primal_coords + (0.45 if is_lowest else 0.5)),
                ucid=d, solid=is_lowest
            ))
        return wire_blocks

    @classmethod
    def joined(cls, svhs: List["SparseFeatureHierarchy"], transforms: List[Isometry]):
        ref_svh = svhs[0]
        inst = cls(ref_svh.voxel_size, ref_svh.depth, ref_svh.device)
        for d in range(ref_svh.depth):
            d_samples = []
            vs, orig = inst.get_grid_voxel_size_origin(d)
            for svh, iso in zip(svhs, transforms):
                grid = svh.grids[d]
                if grid is None:
                    continue
                test_pos = torch.linspace(-0.5, 0.5, 3, device=ref_svh.device)
                test_pos = torch.stack(torch.meshgrid(test_pos, test_pos, test_pos, indexing='ij'), dim=3)
                test_pos = test_pos.view(-1, 3) * 0.99
                test_pos = grid.active_grid_coords().unsqueeze(1) + test_pos.unsqueeze(0)
                test_pos = iso @ grid.grid_to_world(test_pos.view(-1, 3))
                test_pos = ((test_pos - orig) / vs).round().long()
                test_pos = torch.unique(test_pos, dim=0)
                d_samples.append(test_pos)
            d_samples = torch.unique(torch.cat(d_samples, dim=0), dim=0)
            inst.build_from_grid_coords(d, d_samples)
        return inst

    def permute_features(self, depth: int, features: torch.Tensor, out_coords: torch.Tensor, strict: bool = True):
        grid = self.grids[depth]
        assert grid is not None, "grid structure is not built!"
        assert features.size(0) == grid.num_voxels, "Feature size does not match num voxels!"
        vidx = grid.ijk_to_index(out_coords)

        if strict:
            assert out_coords.size(0) == features.size(0), "out coords size does not match!"
            vidx_sorted = torch.sort(vidx)[0]
            assert torch.all(vidx_sorted == torch.arange(grid.num_voxels, dtype=vidx.dtype, device=vidx.device))
            return features[vidx]

        else:
            out_size = list(features.shape)
            out_size[0] = out_coords.size(0)
            out_feat = torch.zeros(out_size, device=features.device, dtype=features.dtype)
            in_mask = vidx != -1
            out_feat[in_mask] = features[vidx[in_mask]]
            return out_feat

    def to_(self, device: Device):
        device = torch.device(device)
        if device == self.device:
            return
        self.device = device
        self.kernel_maps = {k: v.to(device) for k, v in self.kernel_maps.items()}
        self.grids = [v.to(device) if v is not None else None for v in self.grids]

    def clear_kernel_maps(self):
        self.kernel_maps = {}

    def build_iterative_coarsening(self, pts: torch.Tensor):
        assert pts.device == self.device, f"Device not match {pts.device} vs {self.device}."
        self.grids[0] = SparseIndexGrid(*self.get_grid_voxel_size_origin(0), device=self.device)
        self.grids[0].build_from_pointcloud(pts, [0, 0, 0], [0, 0, 0])
        for d in range(1, self.depth):
            self.grids[d] = self.grids[d - 1].coarsened_grid(2)

    def build_point_splatting(self, pts: torch.Tensor):
        assert pts.device == self.device, f"Device not match {pts.device} vs {self.device}."
        for d in range(self.depth):
            self.grids[d] = SparseIndexGrid(*self.get_grid_voxel_size_origin(d), device=self.device)
            self.grids[d].build_from_pointcloud_nearest_voxels(pts)

    def build_adaptive_normal_variation(self, pts: torch.Tensor, normal: torch.Tensor,
                                        tau: float = 0.2, adaptive_depth: int = 100):
        assert pts.device == normal.device == self.device, "Device not match"
        inv_mapping = None
        for d in range(self.depth - 1, -1, -1):
            # Obtain points & normals for this level
            if inv_mapping is not None:
                nx, ny, nz = torch.abs(normal[:, 0]), torch.abs(normal[:, 1]), torch.abs(normal[:, 2])
                vnx = torch_scatter.scatter_std(nx, inv_mapping)
                vny = torch_scatter.scatter_std(ny, inv_mapping)
                vnz = torch_scatter.scatter_std(nz, inv_mapping)
                pts_mask = ((vnx + vny + vnz) > tau)[inv_mapping]
                pts, normal = pts[pts_mask], normal[pts_mask]

            if pts.size(0) == 0:
                return

            self.grids[d] = SparseIndexGrid(*self.get_grid_voxel_size_origin(d), device=self.device)
            self.grids[d].build_from_pointcloud_nearest_voxels(pts)

            if 0 < d < adaptive_depth:
                inv_mapping = self.grids[d].ijk_to_index(self.grids[d].world_to_grid(pts).round().int())

    def build_from_grid_coords(self, depth: int, grid_coords: torch.Tensor,
                               pad_min: list = None, pad_max: list = None):
        if pad_min is None:
            pad_min = [0, 0, 0]

        if pad_max is None:
            pad_max = [0, 0, 0]

        assert grid_coords.device == self.device, "Device not match"
        assert self.grids[depth] is None, "Grid is not empty"
        self.grids[depth] = SparseIndexGrid(*self.get_grid_voxel_size_origin(depth), device=self.device)
        self.grids[depth].build_from_ijk_coords(grid_coords, pad_min, pad_max)

    def build_from_grid(self, depth: int, grid: SparseIndexGrid):
        assert self.grids[depth] is None, "Grid is not empty"
        grid_size, grid_origin = self.get_grid_voxel_size_origin(depth)
        assert grid.voxel_size == grid_size, f"Voxel size does not match: {grid.voxel_size} vs {grid_size}!"
        assert grid.origin == grid_origin, f"Origin does not match: {grid.origin} vs {grid_origin}!"
        self.grids[depth] = grid
