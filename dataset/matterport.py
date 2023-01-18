from pathlib import Path

import numpy as np

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms


class MatterportDataset(RandomSafeDataset):
    def __init__(self, base_path, spec, split, tv_ratio=0.7/0.8, transforms=None, partial_input=False,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="matterport", custom_scenes=None,
                 **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name

        assert DS.GT_MESH not in spec and DS.GT_MESH_SOUP not in spec
        self.split = split
        self.spec = self.sanitize_specs(
            spec, [DS.SHAPE_NAME, DS.INPUT_PC, DS.TARGET_NORMAL, DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL])
        self.transforms = ComposedTransforms(transforms)
        self.base_path = Path(base_path)

        if self.split == "test":
            with (self.base_path / "scenes_test.txt").open() as f:
                self.scenes = [t.strip() for t in f.readlines()]
        elif self.split == "custom":
            assert custom_scenes is not None
            self.scenes = custom_scenes
        else:
            with (self.base_path / "scenes_train.txt").open() as f:
                all_scenes = [t.strip() for t in f.readlines()]
            np.random.RandomState(0).shuffle(all_scenes)
            n_train = int(len(all_scenes) * tv_ratio)
            if self.split == "train":
                self.scenes = all_scenes[:n_train]
            elif self.split == "val":
                self.scenes = all_scenes[n_train:]

        # Get all models
        self.regions = []
        for scene in self.scenes:
            for region in sorted(list((self.base_path / scene).glob('*'))):
                self.regions.append({'scene': scene, 'region': region.name})
        self.hparams = hparams
        self.partial_input = partial_input

    def __len__(self):
        return len(self.regions)

    def get_name(self):
        return f"{self.custom_name}-{self.split}"

    def get_short_name(self):
        return f"{self.custom_name}"

    def _get_item(self, data_id, rng):
        scene = self.regions[data_id]['scene']
        region = self.regions[data_id]['region']

        data = {}

        full_data = np.load(self.base_path / scene / region / "full.npz")
        full_points = full_data['points'].astype(np.float32)
        full_normals = full_data['normals'].astype(np.float32)

        if self.partial_input:
            partial_data = np.load(self.base_path / scene / region / "partial.npz")
            partial_points = partial_data['points'].astype(np.float32)
            partial_normals = partial_data['normals'].astype(np.float32)
        else:
            partial_points, partial_normals = np.copy(full_points), np.copy(full_normals)

        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = "/".join([scene, region])

        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = full_points

        if DS.GT_DENSE_NORMAL in self.spec:
            data[DS.GT_DENSE_NORMAL] = full_normals

        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = partial_points

        if DS.TARGET_NORMAL in self.spec:
            data[DS.TARGET_NORMAL] = partial_normals

        if self.transforms is not None:
            data = self.transforms(data, rng)

        return data
