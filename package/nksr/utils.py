# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import itertools

import torch
import torch_scatter
import numpy as np
from nksr import ext

from pycg.isometry import Isometry
from collections import defaultdict
from typing import Union


Device = Union[torch.device, str]


def get_device(device: Device):
    device = torch.device(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if device.type == 'cuda':
        device = torch.device(device.type, device_index)
    elif device.type == 'cpu':
        device = torch.device(device.type)
    else:
        raise ValueError("Only supports 'cpu' and 'cuda' devices")
    return device


def subdivide_cube_indices(cube_graph: torch.Tensor, cube_vertices: torch.Tensor):
    cube_graph = [cube_graph[:, idx] for idx in range(8)]
    new_cube_graph = [[None for _ in range(8)] for _ in range(8)]

    new_cube_vertices = [cube_vertices]
    new_cube_n_vert = cube_vertices.size(0)

    # Populate vertex vertices
    for vidx in range(8):
        new_cube_graph[vidx][vidx] = cube_graph[vidx]

    # Add edge vertices (vertical, horizontal, depth)
    for edge_idx_set in [[[0, 4], [1, 5], [3, 7], [2, 6]],
                         [[0, 1], [3, 2], [4, 5], [7, 6]],
                         [[0, 3], [1, 2], [4, 7], [5, 6]]]:
        # Find all unique edges to add vertex
        edge_v_idx = []
        for edge_idx in edge_idx_set:
            edge_v_idx.append(torch.stack([cube_graph[edge_idx[0]], cube_graph[edge_idx[1]]], dim=1))
        edge_v_idx = torch.cat(edge_v_idx, dim=0)
        edge_v_idx, edge_inv_mapping = torch.unique(edge_v_idx, dim=0, return_inverse=True)
        edge_vertices = (cube_vertices[edge_v_idx[:, 0]] + cube_vertices[edge_v_idx[:, 1]]) / 2.

        # For collapsed edges don't add vertex
        deg_mask = edge_v_idx[:, 0] != edge_v_idx[:, 1]
        edge_vertices = edge_vertices[deg_mask]
        new_cube_vertices.append(edge_vertices)

        # Determine indexing for the new edges
        edge_new_idx = edge_v_idx[:, 0]
        edge_new_idx[deg_mask] = torch.arange(
            edge_vertices.size(0), device=cube_vertices.device, dtype=torch.long) + new_cube_n_vert
        new_cube_n_vert += edge_vertices.size(0)

        # Assign new vertex to new cubes
        edge_new_idx = edge_new_idx[edge_inv_mapping]
        edge_new_idx = torch.chunk(edge_new_idx, 4)
        for group_idx, edge_idx in enumerate(edge_idx_set):
            new_cube_graph[edge_idx[0]][edge_idx[1]] = edge_new_idx[group_idx]
            new_cube_graph[edge_idx[1]][edge_idx[0]] = edge_new_idx[group_idx]

    # Add face vertices
    for face_idx_set in [[[0, 1, 5, 4], [3, 2, 6, 7]],
                         [[1, 2, 6, 5], [0, 3, 7, 4]],
                         [[0, 1, 2, 3], [4, 5, 6, 7]]]:
        # Find all unique faces to add vertex
        face_v_idx = []
        for face_idx in face_idx_set:
            face_v_idx.append(torch.stack([cube_graph[face_idx[i]] for i in range(4)], dim=1))
        face_v_idx = torch.cat(face_v_idx, dim=0)
        face_v_idx, face_inv_mapping = torch.unique(face_v_idx, dim=0, return_inverse=True)
        face_vertices = sum([cube_vertices[face_v_idx[:, i]] for i in range(4)]) / 4.

        # For collapsed faces don't add vertex
        deg_mask = ~((face_v_idx[:, 0] == face_v_idx[:, 1]) &
                     (face_v_idx[:, 0] == face_v_idx[:, 2]) &
                     (face_v_idx[:, 0] == face_v_idx[:, 3]))
        face_vertices = face_vertices[deg_mask]
        new_cube_vertices.append(face_vertices)

        # Determine indexing for the new faces
        face_new_idx = face_v_idx[:, 0]
        face_new_idx[deg_mask] = torch.arange(
            face_vertices.size(0), device=cube_vertices.device, dtype=torch.long) + new_cube_n_vert
        new_cube_n_vert += face_vertices.size(0)

        # Assign new vertex to new cubes
        face_new_idx = face_new_idx[face_inv_mapping]
        face_new_idx = torch.chunk(face_new_idx, 2)
        for group_idx, face_idx in enumerate(face_idx_set):
            for i in range(4):
                new_cube_graph[face_idx[i]][face_idx[(i + 2) % 4]] = face_new_idx[group_idx]

    # Add center vertices
    center_vertices = sum([cube_vertices[cube_graph[i]] for i in range(8)]) / 8.
    new_cube_vertices.append(center_vertices)
    center_new_idx = torch.arange(
        center_vertices.size(0), device=cube_vertices.device, dtype=torch.long) + new_cube_n_vert
    new_cube_n_vert += center_vertices.size(0)
    for cur_idx, diag_idx in enumerate([6, 7, 4, 5, 2, 3, 0, 1]):
        new_cube_graph[cur_idx][diag_idx] = center_new_idx

    new_cube_graph = torch.cat([torch.stack(new_cube_graph[i], dim=1) for i in range(8)])
    new_cube_vertices = torch.cat(new_cube_vertices)
    return new_cube_graph, new_cube_vertices


def apply_vertex_mask(mesh_v: torch.Tensor, mesh_f: torch.Tensor, v_mask: torch.Tensor):
    assert mesh_v.size(0) == v_mask.size(0), "size not match when applying vertex masks!"
    vert_idx_mapping = torch.full((mesh_v.size(0), ), -1, dtype=torch.long, device=mesh_v.device)
    vert_idx_mapping[v_mask] = torch.arange(torch.sum(v_mask), dtype=torch.long, device=mesh_v.device)
    mesh_f = vert_idx_mapping[mesh_f]
    mesh_v = mesh_v[v_mask]
    mesh_f = mesh_f[torch.all(mesh_f != -1, dim=1)]
    return mesh_v, mesh_f


def points_voxel_downsample(xyz: torch.Tensor, voxel_size: float):
    xyz_inds = torch.div(xyz, voxel_size, rounding_mode='floor')
    _, xyz_inds = torch.unique(xyz_inds, dim=0, return_inverse=True)
    return torch_scatter.scatter_mean(xyz, xyz_inds, dim=0)


def split_into_chunks(xyz: torch.Tensor,
                      chunk_size: float, overlap_ratio: float,
                      **features):

    # Determine extent of the scene
    extent_min, extent_max = torch.min(xyz, dim=0).values, torch.max(xyz, dim=0).values
    extent_min = extent_min.cpu().numpy()
    extent_max = extent_max.cpu().numpy()
    extent_size = extent_max - extent_min
    extent_center = (extent_min + extent_max) / 2.
    extent_min = extent_min - extent_size * 0.05
    extent_max = extent_max + extent_size * 0.05
    extent_size = extent_max - extent_min

    # Determine chunk division, and recompute bound.
    chunk_sub_size = chunk_size * (1 - overlap_ratio)
    n_extent = np.ceil((extent_size - chunk_size * overlap_ratio) / chunk_sub_size)
    extent_size = n_extent * chunk_sub_size + chunk_size * overlap_ratio
    extent_min = extent_center - extent_size / 2.
    extent_max = extent_center + extent_size / 2.

    chunk_xyzs, chunk_features, chunk_transforms = [], defaultdict(list), []
    for nx, ny, nz in itertools.product(
            np.arange(n_extent[0]), np.arange(n_extent[1]), np.arange(n_extent[2])):

        chunk_min = extent_min + np.asarray([nx, ny, nz]) * chunk_sub_size
        chunk_max = chunk_min + np.asarray([chunk_size] * 3)
        chunk_iso = Isometry(t=(chunk_min + chunk_max) / 2.)
        chunk_min = torch.from_numpy(chunk_min).float().to(xyz.device)
        chunk_max = torch.from_numpy(chunk_max).float().to(xyz.device)

        pts_mask = torch.logical_and(
            torch.all(xyz > chunk_min[None, :], dim=1),
            torch.all(xyz < chunk_max[None, :], dim=1))
        if not torch.any(pts_mask):
            continue

        chunk_transforms.append(chunk_iso)
        chunk_xyzs.append(chunk_iso.inv() @ xyz[pts_mask])

        for feat_name, feat_value in features.items():
            if feat_value is None:
                chunk_features[feat_name].append(None)
            elif feat_name == 'normal':
                chunk_features[feat_name].append(
                    chunk_iso.rotation.inv() @ feat_value[pts_mask]
                )
            elif feat_name == 'sensor':
                chunk_features[feat_name].append(
                    chunk_iso.inv() @ feat_value[pts_mask]
                )
            else:
                raise NotImplementedError

    return chunk_transforms, chunk_xyzs, chunk_features


def filter_radius_inliers(xyz: torch.Tensor, knn: int, outlier_radius: float):
    """
    Remove points too far away from the potential geometry
    :param xyz: (N, 3) torch.Tensor
    :param knn: int
    :param outlier_radius: float, point whose (knn-1)th neighbour with distance larger than this will be discarded
    :return: indices (M, )
    """
    torch.cuda.empty_cache()
    knn_dist, knn_indices = ext.pcproc.nearest_neighbours(xyz, knn)
    inlier_mask = knn_dist[:, -1] <= (outlier_radius * outlier_radius)
    return torch.where(inlier_mask)[0]


def estimate_normals(xyz: torch.Tensor, sensor: torch.Tensor,
                     knn: int,
                     drop_threshold_degrees: float = 90.0,
                     backend: str = 'nksr'):
    """
    Estimate normals of a point cloud, orientation determined by sensor positions.
    :param xyz: (N, 3) torch.Tensor
    :param sensor: (N, 3) sensor positions
    :param knn: int, used for normal estimation.
    :param drop_threshold_degrees: float, angle(ray-dir, normal) larger than this threshold will be discarded
    :param backend: str
    :return: normal (M, 3), indices (M, )
    """
    assert xyz.size(0) == sensor.size(0)
    assert backend in ['nksr', 'open3d', 'pcu'], "backend not supported!"

    if backend == 'nksr':
        indices = torch.arange(xyz.size(0), dtype=torch.long, device=xyz.device)

        torch.cuda.empty_cache()
        knn_dist, knn_indices = ext.pcproc.nearest_neighbours(xyz, knn)
        normal = ext.pcproc.estimate_normals_knn(xyz, knn_dist, knn_indices)

        # (Below required only when max_radius is specified)
        # normal_valid_mask = ~torch.any(torch.isnan(normal), dim=1)
        # xyz, sensor, indices = xyz[normal_valid_mask], sensor[normal_valid_mask], indices[normal_valid_mask]
        # normal = normal[normal_valid_mask]

        del knn_dist, knn_indices

    elif backend == 'pcu':
        import point_cloud_utils as pcu

        xyz_numpy = xyz.cpu().numpy()
        indices, normal = pcu.estimate_point_cloud_normals_knn(xyz_numpy, knn)
        normal = torch.from_numpy(normal).to(xyz)
        indices = torch.from_numpy(indices).to(xyz)

        xyz, sensor = xyz[indices], sensor[indices]

    elif backend == 'open3d':
        import open3d as o3d

        indices = torch.arange(xyz.size(0), dtype=torch.long, device=xyz.device)

        device = o3d.core.Device(
            o3d.core.Device.CUDA if xyz.is_cuda else o3d.core.Device.CPU,
            xyz.device.index
        )
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(xyz))
        pcd.estimate_normals(knn)
        normal = torch.utils.dlpack.from_dlpack(pcd.point.normals.to_dlpack()).clone()

    else:
        raise NotImplementedError

    view_dir = sensor - xyz
    view_dir = view_dir / (torch.linalg.norm(view_dir, dim=-1, keepdim=True) + 1e-6)
    cos_angle = torch.sum(view_dir * normal, dim=1)
    cos_mask = cos_angle < 0.0
    normal[cos_mask] = -normal[cos_mask]
    del cos_mask

    if drop_threshold_degrees < 90.0:
        keep_mask = torch.abs(cos_angle) > np.cos(np.deg2rad(drop_threshold_degrees))
        xyz, normal, indices = xyz[keep_mask], normal[keep_mask], indices[keep_mask]

    return normal, indices
