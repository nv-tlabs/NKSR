import os
from pathlib import Path

import numpy as np

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms


class ShapeNetDataset(RandomSafeDataset):
    def __init__(self, onet_base_path, spec, split,
                 onet_color_path=None, shapenet_base_path=None, categories=None, transforms=None,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="shapenet", use_dummy_iou=False,
                 **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name

        # For data packs without points.npz, return a dummy occupancy.
        self.use_dummy_iou = use_dummy_iou

        if DS.GT_MESH in spec or DS.GT_MESH_SOUP in spec:
            assert shapenet_base_path is not None
            self.shapenet_base_path = Path(shapenet_base_path)
        else:
            self.shapenet_base_path = None

        if DS.INPUT_COLOR in spec or DS.GT_DENSE_COLOR in spec:
            assert onet_color_path is not None
            self.onet_color_path = Path(onet_color_path)
        else:
            self.onet_color_path = None

        self.split = split
        self.spec = self.sanitize_specs(
            spec, [DS.SHAPE_NAME, DS.INPUT_PC, DS.TARGET_NORMAL, DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL, DS.GT_ONET_SAMPLE,
                   DS.INPUT_COLOR, DS.GT_DENSE_COLOR])
        self.transforms = ComposedTransforms(transforms)

        # If categories is None, use all sub-folders
        if categories is None:
            base_path = Path(onet_base_path)
            categories = os.listdir(base_path)
            categories = [c for c in categories if (base_path / c).is_dir()]
        self.categories = categories

        # Get all models
        self.models = []
        self.onet_base_paths = {}
        for c in categories:
            self.onet_base_paths[c] = Path(onet_base_path + "/" + c)
            split_file = self.onet_base_paths[c] / (split + '.lst')
            with split_file.open('r') as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.models += [{'category': c, 'model': m} for m in models_c]
        self.hparams = hparams

    def __len__(self):
        return len(self.models)

    def get_name(self):
        return f"{self.custom_name}-cat{len(self.categories)}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        category = self.models[data_id]['category']
        model = self.models[data_id]['model']

        data = {}

        gt_data = np.load(self.onet_base_paths[category] / model / 'pointcloud.npz')
        gt_points = gt_data['points'].astype(np.float32)
        gt_normals = gt_data['normals'].astype(np.float32)

        # Load color
        if self.onet_color_path is not None:
            gt_color = np.load(self.onet_color_path / category / model / 'color.npz')['rgb']
            for key in [DS.INPUT_COLOR, DS.GT_DENSE_COLOR]:
                if key in self.spec:
                    data[key] = gt_color.astype(np.float32)

        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = "/".join([category, model])

        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = gt_points

        if DS.GT_DENSE_NORMAL in self.spec:
            data[DS.GT_DENSE_NORMAL] = gt_normals

        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = gt_points

        if DS.TARGET_NORMAL in self.spec:
            data[DS.TARGET_NORMAL] = gt_normals

        if DS.GT_MESH in self.spec or DS.GT_MESH_SOUP in self.spec:
            import open3d as o3d
            origin_mesh_path = self.shapenet_base_path / category / model / 'model.obj'
            origin_mesh = o3d.io.read_triangle_mesh(str(origin_mesh_path))
            origin_mesh.scale(1.0 / gt_data['scale'], center=[0.0, 0.0, 0.0])
            origin_mesh.translate(-gt_data['loc'] / gt_data['scale'])
            if DS.GT_MESH in self.spec:
                data[DS.GT_MESH] = origin_mesh
            if DS.GT_MESH_SOUP in self.spec:
                verts, tris = np.asarray(origin_mesh.vertices).astype(np.float32), np.asarray(origin_mesh.triangles)
                data[DS.GT_MESH_SOUP] = np.stack([verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]], axis=1)

        # points and occupancy samples used to train O-Net
        if DS.GT_ONET_SAMPLE in self.spec:
            if self.use_dummy_iou:
                sample_points = np.zeros((32, 3), dtype=np.float32)
                sample_occ = np.zeros((32, ), dtype=bool)
            else:
                samples = np.load(self.onet_base_paths[category] / model / "points.npz")
                sample_points = samples['points'].astype(np.float32)
                sample_occ = np.unpackbits(samples['occupancies'])[:sample_points.shape[0]]
            data[DS.GT_ONET_SAMPLE] = [sample_points, sample_occ]

        # (not needed) 32^3 binary voxel data
        # with open(os.path.join(args.onet_base_path, c, obj_name, "model.binvox"), "rb") as fp:
        #     voxel = binvox_util.read_as_3d_array(fp)

        if self.transforms is not None:
            data = self.transforms(data, rng)

        return data
