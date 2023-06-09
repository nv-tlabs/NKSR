import os
from pathlib import Path

import numpy as np

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms
from dataset.av_gt_geometry import get_class
from pycg import exp


class AVDataset(RandomSafeDataset):
    def __init__(self, base_path, spec, split, input_path=None, drives=None, transforms=None,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="unnamed-av",
                 use_dummy_gt=False, **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name

        self.split = split
        self.spec = self.sanitize_specs(spec, [DS.SHAPE_NAME, DS.INPUT_PC, DS.TARGET_NORMAL, DS.GT_GEOMETRY, DS.INPUT_SENSOR_POS])
        self.transforms = ComposedTransforms(transforms)
        self.use_dummy_gt = use_dummy_gt

        # If drives not specified, use all sub-folders
        base_path = Path(base_path)
        if drives is None:
            drives = os.listdir(base_path)
            drives = [c for c in drives if (base_path / c).is_dir()]
        self.drives = drives
        self.input_path = input_path

        # Get all items
        self.all_items = []
        self.drive_base_paths = {}
        for c in drives:
            self.drive_base_paths[c] = base_path / c
            split_file = self.drive_base_paths[c] / (split + '.lst')
            with split_file.open('r') as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.all_items += [{'drive': c, 'item': m} for m in models_c]
        self.hparams = hparams

    def __len__(self):
        return len(self.all_items)

    def get_name(self):
        return f"{self.custom_name}-cat{len(self.drives)}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        drive_name = self.all_items[data_id]['drive']
        item_name = self.all_items[data_id]['item']

        data = {}

        try:
            if self.input_path is None:
                input_data = np.load(self.drive_base_paths[drive_name] / item_name / 'pointcloud.npz')
            else:
                input_data = np.load(Path(self.input_path) / drive_name / item_name / 'pointcloud.npz')
        except FileNotFoundError:
            exp.logger.warning(f"File not found for AV dataset for {item_name}")
            raise ConnectionAbortedError

        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = "/".join([drive_name, item_name])

        if DS.INPUT_PC in self.spec:
            input_points = input_data['points'].astype(np.float32)
            data[DS.INPUT_PC] = input_points

        if DS.TARGET_NORMAL in self.spec:
            input_normals = input_data['normals'].astype(np.float32)
            data[DS.TARGET_NORMAL] = input_normals

        if DS.INPUT_SENSOR_POS in self.spec:
            input_sensor = input_data['sensor'].astype(np.float32)
            data[DS.INPUT_SENSOR_POS] = input_sensor

        if DS.GT_GEOMETRY in self.spec:
            geom_cls = get_class(self.hparams.supervision.gt_type)
            if self.use_dummy_gt:
                data[DS.GT_GEOMETRY] = geom_cls.empty()
            else:
                data[DS.GT_GEOMETRY] = geom_cls.load(self.drive_base_paths[drive_name] / item_name / "groundtruth.bin")

        if self.transforms is not None:
            data = self.transforms(data, rng)

        return data
