import numpy as np
import torch
from pathlib import Path
from pycg.exp import lru_cache_class, logger

from pycg.isometry import Isometry
import torch.nn.functional as F


class AVGroundTruthGeometry:
    def __init__(self):
        pass

    @classmethod
    def load(cls, path: Path):
        raise NotImplementedError

    def save(self, path: Path):
        raise NotImplementedError

    def crop(self, bounds: np.ndarray):
        # bounds: (C, 2, 3) min_coords and max_coords
        raise NotImplementedError

    def transform(self, iso: Isometry = Isometry(), scale: float = 1.0):
        # p <- s(Rp+t)
        raise NotImplementedError


class DensePointsGroundTruthGeometry(AVGroundTruthGeometry):
    def __init__(self, xyz: np.ndarray, normal: np.ndarray):
        super().__init__()
        self.xyz = xyz
        self.normal = normal
        assert self.xyz.shape[0] == self.normal.shape[0]
        assert self.xyz.shape[1] == self.normal.shape[1] == 3

    def save(self, path: Path):
        with path.open("wb") as f:
            np.savez_compressed(f, xyz=self.xyz, normal=self.normal)

    def transform(self, iso: Isometry = Isometry(), scale: float = 1.0):
        self.xyz = scale * (iso @ self.xyz)
        self.normal = iso.rotation @ self.normal

    def is_empty(self):
        return self.xyz.shape[0] < 64

    @classmethod
    def load(cls, path: Path):
        res = np.load(path, allow_pickle=True)
        inst = cls(res['xyz'], res['normal'])
        return inst

    @classmethod
    def empty(cls):
        return cls(np.zeros((0, 3)), np.zeros((0, 3)))

    @lru_cache_class(maxsize=None)
    def torch_attr(self):
        return torch.from_numpy(self.xyz).float().cuda(), torch.from_numpy(self.normal).float().cuda()

    def query_sdf(self, queries: torch.Tensor):
        import ext
        all_points_torch, all_normals_torch = self.torch_attr()

        sdf_kwargs = {
            'queries': queries, 'ref_xyz': all_points_torch, 'ref_normal': all_normals_torch,
            'nb_points': 8, 'stdv': 3.0, 'adaptive_knn': 8
        }
        try:
            query_sdf = -ext.sdfgen.sdf_from_points(**sdf_kwargs)[0]
        except MemoryError:
            logger.warning("Query SDF OOM. Try empty pytorch cache.")
            torch.cuda.empty_cache()
            query_sdf = -ext.sdfgen.sdf_from_points(**sdf_kwargs)[0]

        return query_sdf

    def crop(self, bounds: np.ndarray):
        crops = []
        for cur_bound in bounds:
            min_bound, max_bound = cur_bound[0], cur_bound[1]
            crop_mask = np.logical_and.reduce([
                self.xyz[:, 0] > min_bound[0], self.xyz[:, 0] < max_bound[0],
                self.xyz[:, 1] > min_bound[1], self.xyz[:, 1] < max_bound[1],
                self.xyz[:, 2] > min_bound[2], self.xyz[:, 2] < max_bound[2]
            ])
            crop_inst = self.__class__(self.xyz[crop_mask], self.normal[crop_mask])
            crops.append(crop_inst)
        return crops


class PointTSDFVolumeGroundTruthGeometry(AVGroundTruthGeometry):
    def __init__(self, dense_points: DensePointsGroundTruthGeometry,
                 volume: np.ndarray, volume_min: np.ndarray, volume_max: np.ndarray):
        super().__init__()
        self.dense_points = dense_points
        self.volume = volume
        self.volume_min = volume_min
        self.volume_max = volume_max
        assert np.all(self.volume_min < self.volume_max)

    @property
    def xyz(self):
        return self.dense_points.xyz

    @property
    def normal(self):
        return self.dense_points.normal

    @classmethod
    def empty(cls):
        return cls(DensePointsGroundTruthGeometry.empty(), np.ones((1, 1, 1)), np.zeros(3,), np.ones(3,))

    def is_empty(self):
        return self.dense_points.is_empty()

    def save(self, path: Path):
        with path.open("wb") as f:
            np.savez_compressed(f, xyz=self.dense_points.xyz, normal=self.dense_points.normal,
                                volume=self.volume,
                                volume_min=self.volume_min, volume_max=self.volume_max)

    def transform(self, iso: Isometry = Isometry(), scale: float = 1.0):
        assert iso.q.is_unit(), "Volume transform does not support rotation yet"
        self.dense_points.transform(iso, scale)
        self.volume_min = scale * (self.volume_min + iso.t)
        self.volume_max = scale * (self.volume_max + iso.t)

    @classmethod
    def load(cls, path: Path):
        dense_points = DensePointsGroundTruthGeometry.load(path)
        res = np.load(path)
        return cls(dense_points, res['volume'], res['volume_min'], res['volume_max'])

    @lru_cache_class(maxsize=None)
    def torch_attr(self):
        return *self.dense_points.torch_attr(), torch.from_numpy(self.volume).float().cuda()

    def query_classification(self, queries: torch.Tensor, band: float = 1.0):
        """
        Return integer classifications of the query points:
            0 - near surface
            1 - far surface empty
            2 - unknown (also for query points outside volume)
        :param queries: torch.Tensor (N, 3)
        :param band: 0-1 band size to be classified as 'near-surface'
        :return: (N, ) ids
        """
        _, _, volume_input = self.torch_attr()

        in_volume_mask = (queries[:, 0] >= self.volume_min[0]) & (queries[:, 0] <= self.volume_max[0]) & \
                         (queries[:, 1] >= self.volume_min[1]) & (queries[:, 1] <= self.volume_max[1]) & \
                         (queries[:, 2] >= self.volume_min[2]) & (queries[:, 2] <= self.volume_max[2])

        queries_norm = queries[in_volume_mask].clone()
        for i in range(3):
            queries_norm[:, i] = (queries_norm[:, i] - self.volume_min[i]) / \
                                 (self.volume_max[i] - self.volume_min[i]) * 2. - 1.
        sample_grid = torch.fliplr(queries_norm)[None, None, None, ...]
        # B=1,C=1,Di=1,Hi=1,Wi x B=1,Do=1,Ho=1,Wo,3 --> B=1,C=1,Do=1,Ho=1,Wo
        sample_res = F.grid_sample(volume_input[None, None, ...], sample_grid,
                                   mode='nearest', padding_mode='border', align_corners=True)[0, 0, 0, 0]

        cls_in_volume = torch.ones_like(sample_res, dtype=torch.long)
        cls_in_volume[~torch.isfinite(sample_res)] = 2
        cls_in_volume[torch.abs(sample_res) < band] = 0

        cls = torch.ones(queries.size(0), dtype=torch.long, device=cls_in_volume.device) * 2
        cls[in_volume_mask] = cls_in_volume

        return cls

    def query_sdf(self, queries: torch.Tensor):
        return self.dense_points.query_sdf(queries)

    def crop(self, bounds: np.ndarray):
        point_crops = self.dense_points.crop(bounds)

        volume_x_ticks = np.linspace(self.volume_min[0], self.volume_max[0], self.volume.shape[0])
        volume_y_ticks = np.linspace(self.volume_min[1], self.volume_max[1], self.volume.shape[1])
        volume_z_ticks = np.linspace(self.volume_min[2], self.volume_max[2], self.volume.shape[2])

        crops = []
        for cur_point_crop, cur_bound in zip(point_crops, bounds):
            min_bound, max_bound = cur_bound[0], cur_bound[1]
            # volume_ticks[id_min] <= min_bound < max_bound <= volume_ticks[id_max]
            x_id_min = np.maximum(np.searchsorted(volume_x_ticks, min_bound[0], side='right') - 1, 0)
            x_id_max = np.minimum(np.searchsorted(volume_x_ticks, max_bound[0], side='left'),
                                  volume_x_ticks.shape[0] - 1)
            y_id_min = np.maximum(np.searchsorted(volume_y_ticks, min_bound[1], side='right') - 1, 0)
            y_id_max = np.minimum(np.searchsorted(volume_y_ticks, max_bound[1], side='left'),
                                  volume_y_ticks.shape[0] - 1)
            z_id_min = np.maximum(np.searchsorted(volume_z_ticks, min_bound[2], side='right') - 1, 0)
            z_id_max = np.minimum(np.searchsorted(volume_z_ticks, max_bound[2], side='left'),
                                  volume_z_ticks.shape[0] - 1)
            crops.append(self.__class__(
                cur_point_crop,
                self.volume[x_id_min:x_id_max+1, y_id_min:y_id_max+1, z_id_min:z_id_max+1],
                np.array([volume_x_ticks[x_id_min], volume_y_ticks[y_id_min], volume_z_ticks[z_id_min]]),
                np.array([volume_x_ticks[x_id_max], volume_y_ticks[y_id_max], volume_z_ticks[z_id_max]])
            ))
        return crops


def get_class(class_name):
    if class_name == "DensePoints":
        return DensePointsGroundTruthGeometry
    elif class_name == "PointTSDFVolume":
        return PointTSDFVolumeGroundTruthGeometry
    else:
        raise NotImplementedError
