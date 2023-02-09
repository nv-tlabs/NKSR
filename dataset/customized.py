from pathlib import Path
import open3d as o3d

import numpy as np
from pycg.exp import logger

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms


class CustomizedDataset(RandomSafeDataset):

    def __init__(self, data, spec, transforms=None, random_seed=0, hparams=None, skip_on_error=False,
                 custom_name='various', **kwargs):

        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.transforms = ComposedTransforms(transforms)
        self.hparams = hparams
        self.custom_name = custom_name
        self.spec = self.sanitize_specs(spec, [DS.SHAPE_NAME, DS.INPUT_PC, DS.TARGET_NORMAL,
                                               DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL])

        self.data = []
        for datum in data:
            if "input" in datum:
                self.data.append((Path(datum["input"]), Path(datum["gt"])))
            else:
                data_list_name = Path(datum["list"])
                with data_list_name.open("r") as f:
                    content = f.read().strip().split("\n")
                    content = [t.split() for t in content]
                    content = [(data_list_name.parent / t[0], data_list_name.parent / t[1]) for t in content]
                self.data += content
                logger.info(f"Customized dataset parsed list {data_list_name}, containing {len(content)} files.")

    def __len__(self):
        return len(self.data)

    def get_name(self):
        return f"{self.custom_name}-{len(self.data)}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        data = {}

        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = self.data[data_id][0].stem

        if DS.INPUT_PC in self.spec or DS.TARGET_NORMAL in self.spec:
            input_pcd = o3d.io.read_point_cloud(str(self.data[data_id][0]))
            if DS.INPUT_PC in self.spec:
                data[DS.INPUT_PC] = np.asarray(input_pcd.points).astype(np.float32)
            if DS.TARGET_NORMAL in self.spec:
                data[DS.TARGET_NORMAL] = np.asarray(input_pcd.normals).astype(np.float32)

        if DS.GT_DENSE_PC in self.spec or DS.GT_DENSE_NORMAL in self.spec:
            gt_pcd = o3d.io.read_point_cloud(str(self.data[data_id][1]))
            data[DS.GT_DENSE_PC] = np.asarray(gt_pcd.points).astype(np.float32)
            data[DS.GT_DENSE_NORMAL] = np.asarray(gt_pcd.normals).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data, rng)

        return data
