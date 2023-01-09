import open3d as o3d
from pathlib import Path

import numpy as np

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms


class Points2SurfDataset(RandomSafeDataset):
    def __init__(self, base_path, dataset_name, type_name, spec, split, transforms=None,
                 random_seed=0, hparams=None, skip_on_error=False, **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)

        self.skip_on_error = skip_on_error
        self.split = split
        self.spec = self.sanitize_specs(spec, [DS.SHAPE_NAME, DS.INPUT_PC, DS.TARGET_NORMAL,
                                               DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL])
        self.transforms = ComposedTransforms(transforms)
        self.base_path = Path(base_path)
        self.dataset_name = dataset_name
        self.type_name = type_name

        with (self.base_path / self.dataset_name / f"{split}.lst").open('r') as f:
            self.all_items = f.read().strip().split('\n')
        self.all_items = [t for t in self.all_items if len(t) > 0]

        self.hparams = hparams

    def __len__(self):
        return len(self.all_items)

    def get_name(self):
        return f"p2s-{self.dataset_name}-{self.type_name}-{self.split}"

    def get_short_name(self):
        return f"p2s-{self.dataset_name}"

    def _get_item(self, data_id, rng):
        data = {}

        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = self.type_name + "/" + self.all_items[data_id]

        if DS.INPUT_PC in self.spec or DS.TARGET_NORMAL in self.spec:
            input_ply_path = self.base_path / self.dataset_name / self.type_name / \
                             "input" / f"{self.all_items[data_id]}.ply"
            input_pcd = o3d.io.read_point_cloud(str(input_ply_path))
            data[DS.INPUT_PC] = np.asarray(input_pcd.points).astype(np.float32)
            data[DS.TARGET_NORMAL] = np.asarray(input_pcd.normals).astype(np.float32)

        if DS.GT_DENSE_PC in self.spec or DS.GT_DENSE_NORMAL in self.spec:
            gt_ply_path = self.base_path / self.dataset_name / "gt" / f"{self.all_items[data_id]}.ply"
            gt_pcd = o3d.io.read_point_cloud(str(gt_ply_path))
            data[DS.GT_DENSE_PC] = np.asarray(gt_pcd.points).astype(np.float32)
            data[DS.GT_DENSE_NORMAL] = np.asarray(gt_pcd.normals).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data, rng)

        return data
