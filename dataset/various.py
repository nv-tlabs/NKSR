import os
from pathlib import Path
import re
import glob
import open3d as o3d

import numpy as np

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms


class _MeshType:
    def __init__(self, mesh_path, sample_path, sample_args):
        self.mesh_path = mesh_path
        self.sample_path = sample_path
        self.sample_args = sample_args

    @staticmethod
    def parse(data_spec):
        all_specs = []
        mesh_path = data_spec.mesh_path
        mesh_paths = [Path(t) for t in glob.iglob(mesh_path)]
        if "sample_path" in data_spec:
            sample_paths = [Path(data_spec.sample_path.replace('*', t.stem)) for t in mesh_paths]
            for mp, sp in zip(mesh_paths, sample_paths):
                all_specs.append(_MeshType(mp, sp, None))
        else:
            all_specs += [_MeshType(t, None, data_spec.subsample) for t in mesh_paths]
        return all_specs

    def get(self, rng, dataset):
        spec = dataset.spec
        data = {}
        if DS.SHAPE_NAME in spec:
            data[DS.SHAPE_NAME] = self.mesh_path

        if self.sample_path is not None:
            sample_pcd = o3d.io.read_point_cloud(str(self.sample_path))
        else:
            mesh_geom = o3d.io.read_triangle_mesh(str(self.mesh_path))
            mesh_geom.compute_vertex_normals()
            sample_pcd = mesh_geom.sample_points_uniformly(self.sample_args, seed=0)

        if DS.INPUT_PC in spec:
            data[DS.INPUT_PC] = np.array(sample_pcd.points).astype(np.float32)

        if DS.TARGET_NORMAL in spec:
            data[DS.TARGET_NORMAL] = np.array(sample_pcd.normals).astype(np.float32)

        if DS.GT_DENSE_PC in spec:
            data[DS.GT_DENSE_PC] = np.array(sample_pcd.points).astype(np.float32)

        if DS.GT_DENSE_NORMAL in spec:
            data[DS.GT_DENSE_NORMAL] = np.array(sample_pcd.normals).astype(np.float32)

        return data


class _PlyCloudType:
    def __init__(self, path, voxel_down_sample):
        self.path = Path(path)
        self.voxel_down_sample = voxel_down_sample

    @staticmethod
    def parse(data_spec):
        return [_PlyCloudType(data_spec.path, data_spec.get('voxel_down_sample', None))]

    def get(self, rng, dataset):
        spec = dataset.spec
        data = {}
        if DS.SHAPE_NAME in spec:
            data[DS.SHAPE_NAME] = [self.path]

        sample_pcd = o3d.io.read_point_cloud(str(self.path))
        if self.voxel_down_sample is not None:
            sample_pcd = sample_pcd.voxel_down_sample(voxel_size=self.voxel_down_sample)

        if DS.INPUT_PC in spec:
            data[DS.INPUT_PC] = np.array(sample_pcd.points).astype(np.float32)

        if DS.TARGET_NORMAL in spec:
            data[DS.TARGET_NORMAL] = np.array(sample_pcd.normals).astype(np.float32)

        if DS.GT_DENSE_PC in spec:
            data[DS.GT_DENSE_PC] = np.array(sample_pcd.points).astype(np.float32)

        if DS.GT_DENSE_NORMAL in spec:
            data[DS.GT_DENSE_NORMAL] = np.array(sample_pcd.normals).astype(np.float32)

        return data


class _SRBType(_PlyCloudType):
    def __init__(self, base_path, model_name):
        self.base_path = Path(base_path)
        self.model_name = model_name
        super().__init__(self.base_path / "scans" / f"{self.model_name}.ply")

    @staticmethod
    def parse(data_spec):
        all_specs = []
        for model_name in ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]:
            all_specs.append(_SRBType(data_spec.path, model_name))
        return all_specs


class VariousDataset(RandomSafeDataset):
    TYPE_CLS = {
        "mesh": _MeshType,
        "srb": _SRBType,
        "plycloud": _PlyCloudType
    }

    def __init__(self, data, spec, transforms=None, random_seed=0, hparams=None, skip_on_error=False,
                 custom_name='various', **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.transforms = ComposedTransforms(transforms)
        self.spec = spec
        self.data = []
        for d_spec in data:
            self.data += self.TYPE_CLS[d_spec.type].parse(d_spec)
        self.hparams = hparams
        self.custom_name = custom_name

    def __len__(self):
        return len(self.data)

    def get_name(self):
        return f"{self.custom_name}-{len(self.data)}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        data = self.data[data_id].get(rng, self)

        if self.transforms is not None:
            data = self.transforms(data, rng)

        return data
