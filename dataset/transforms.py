import numpy as np
from pycg.isometry import Isometry
from pycg.exp import logger
from numpy.random import RandomState
from dataset.base import DatasetSpec as DS


def pad_cloud(P: np.ndarray, n_in: int, return_inds=False, random_state=None):
    """
    Pad or subsample 3D Point cloud to n_in (fixed) number of points
    :param P: N x C numpy array
    :param n_in: number of points to truncate
    :return: n_in x C numpy array
    """
    if random_state is None:
        random_state = RandomState()

    N = P.shape[0]
    # https://github.com/charlesq34/pointnet/issues/41
    if N > n_in:  # need to subsample
        choice = random_state.choice(N, n_in, replace=False)
    elif N < n_in:  # need to pad by duplication
        ii = random_state.choice(N, n_in - N)
        choice = np.concatenate([range(N), ii])
    else:
        choice = np.arange(N)

    if return_inds:
        return choice
    else:
        return P[choice, :]


class PointcloudNoise:
    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data, rng):
        if self.stddev == 0.0:
            return data
        # Will modify 'INPUT_PC'
        data_out = data.copy()
        if DS.INPUT_PC in data.keys():
            points = data[DS.INPUT_PC]
            noise = self.stddev * rng.randn(*points.shape)
            noise = noise.astype(np.float32)
            data_out[DS.INPUT_PC] = points + noise
        return data_out


class PointcloudOutliers:
    def __init__(self, ratio, spatial_ratio: float = 1.0):
        """
        :param ratio: (float) outlier percentage to the entire point cloud
        :param spatial_ratio: (float) where will be the outliers located -- for each axis the expansion ratio
        """
        self.ratio = ratio
        self.spatial_ratio = spatial_ratio

    def __call__(self, data, rng):
        if self.ratio == 0.0:
            return data

        points = data[DS.INPUT_PC]
        # bound_min, bound_max = np.min(points, axis=0), np.max(points, axis=0)
        # bound_min = bound_min - (bound_max - bound_min) * (self.spatial_ratio / 2.)
        # bound_max = bound_max + (bound_max - bound_min) * (self.spatial_ratio / 2.)
        bound_min, bound_max = -0.55, 0.55

        n_points = points.shape[0]
        n_outlier_points = int(n_points * self.ratio)
        ind = rng.randint(0, n_points, n_outlier_points)

        # Will modify 'INPUT_PC', 'TARGET_NORMAL', 'INPUT_COLOR'
        data_out = data.copy()
        if DS.INPUT_PC in data.keys():
            points = data[DS.INPUT_PC].copy()
            points[ind] = rng.uniform(
                bound_min, bound_max, (n_outlier_points, 3)).astype(np.float32)
            data_out[DS.INPUT_PC] = points

        if DS.TARGET_NORMAL in data.keys():
            normal = data[DS.TARGET_NORMAL].copy()
            random_normal = rng.randn(n_outlier_points, 3)
            normal[ind] = random_normal / np.linalg.norm(random_normal, axis=1, keepdims=True)
            data_out[DS.TARGET_NORMAL] = normal

        if DS.INPUT_COLOR in data.keys():
            color = data[DS.INPUT_COLOR].copy()
            color[ind] = rng.uniform(0.0, 1.0, (n_outlier_points, 3))
            data_out[DS.INPUT_COLOR] = color

        return data_out


class SubsamplePointcloud:
    """ Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): maximum number of points to be subsampled
        n_min (int): minimum number, default is None
    """

    def __init__(self, N, n_min=None):
        self.N = N
        self.n_min = n_min if n_min is not None else N
        assert self.n_min <= self.N

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC' and 'TARGET_NORMAL' and 'INPUT_COLOR'
        data_out = data.copy()
        assert DS.INPUT_PC in data.keys()

        points = data[DS.INPUT_PC]
        if points.shape[0] > self.N:
            indices = pad_cloud(points, self.N, return_inds=True, random_state=rng)
        elif points.shape[0] < self.n_min:
            indices = pad_cloud(points, self.n_min, return_inds=True, random_state=rng)
        else:
            indices = np.arange(points.shape[0])
        data_out[DS.INPUT_PC] = points[indices, :]

        if DS.TARGET_NORMAL in data.keys():
            data_out[DS.TARGET_NORMAL] = data[DS.TARGET_NORMAL][indices, :]

        if DS.INPUT_SENSOR_POS in data.keys():
            data_out[DS.INPUT_SENSOR_POS] = data[DS.INPUT_SENSOR_POS][indices, :]

        if DS.INPUT_COLOR in data.keys():
            data_out[DS.INPUT_COLOR] = data[DS.INPUT_COLOR][indices, :]

        return data_out


class BBoxCrop:
    """
    Use a bbox to crop the points. If points are too few, then re-propose a box before choosing the best.
    """
    def __init__(self, min_l, max_l, low_ratio):
        self.min_l = min_l
        self.max_l = max_l
        self.low_ratio = low_ratio

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC', 'GT_DENSE_PC', 'GT_ONET_SAMPLE' and 'TARGET_NORMAL', 'INPUT_COLOR', 'GT_DENSE_COLOR'
        assert DS.INPUT_PC in data.keys()
        # assert DS.TARGET_NORMAL in data.keys()
        # assert DS.GT_DENSE_PC in data.keys()
        # assert DS.GT_DENSE_NORMAL in data.keys()
        assert DS.GT_GEOMETRY not in data.keys(), "Cannot Apply BBox Crop to DS.GT_GEOMETRY!"

        in_points = data[DS.INPUT_PC]
        in_points_min = np.min(in_points, axis=0)
        in_points_max = np.max(in_points, axis=0)
        in_points_bound = in_points_max - in_points_min
        in_points_min = in_points_min - in_points_bound * 0.1
        in_points_max = in_points_max + in_points_bound * 0.1

        best_ratio = 0.0
        best_inds = None
        best_bounds = None
        for trial_idx in range(20):
            crop_center = rng.uniform(in_points_min, in_points_max)
            crop_size = rng.uniform(self.min_l, self.max_l, (3, ))
            crop_min = np.maximum(crop_center - crop_size / 2., in_points_min)
            crop_max = np.minimum(crop_min + crop_size, in_points_max)
            crop_min = np.maximum(crop_max - crop_size, in_points_min)
            # Apply crop
            pts_inds = np.logical_and.reduce([
                in_points[:, 0] > crop_min[0], in_points[:, 0] < crop_max[0],
                in_points[:, 1] > crop_min[1], in_points[:, 1] < crop_max[1],
                in_points[:, 2] > crop_min[2], in_points[:, 2] < crop_max[2],
            ])
            cur_ratio = np.sum(pts_inds) / pts_inds.shape[0]
            if cur_ratio >= self.low_ratio:
                best_inds = pts_inds
                best_bounds = [crop_min, crop_max]
                break
            else:
                if cur_ratio > best_ratio:
                    best_ratio = cur_ratio
                    best_inds = pts_inds
                    best_bounds = [crop_min, crop_max]

        data_out = data.copy()
        data_out[DS.INPUT_PC] = data[DS.INPUT_PC][best_inds, :]

        if DS.TARGET_NORMAL in data.keys():
            data_out[DS.TARGET_NORMAL] = data[DS.TARGET_NORMAL][best_inds, :]

        if DS.INPUT_SENSOR_POS in data.keys():
            data_out[DS.INPUT_SENSOR_POS] = data[DS.INPUT_SENSOR_POS][best_inds, :]

        if DS.INPUT_COLOR in data.keys():
            data_out[DS.INPUT_COLOR] = data[DS.INPUT_COLOR][best_inds, :]

        if DS.GT_DENSE_PC in data.keys():
            gt_points = data[DS.GT_DENSE_PC]
            gt_inds = np.logical_and.reduce([
                gt_points[:, 0] > best_bounds[0][0], gt_points[:, 0] < best_bounds[1][0],
                gt_points[:, 1] > best_bounds[0][1], gt_points[:, 1] < best_bounds[1][1],
                gt_points[:, 2] > best_bounds[0][2], gt_points[:, 2] < best_bounds[1][2],
            ])
            data_out[DS.GT_DENSE_PC] = gt_points[gt_inds, :]
            if DS.GT_DENSE_NORMAL in data.keys():
                data_out[DS.GT_DENSE_NORMAL] = data[DS.GT_DENSE_NORMAL][gt_inds, :]
            if DS.GT_DENSE_COLOR in data.keys():
                data_out[DS.GT_DENSE_COLOR] = data[DS.GT_DENSE_COLOR][gt_inds, :]

        return data_out


class FixedBBoxCrop:
    """
    Crop the scene using a predefined bound, used for debugging purpose.
    """
    def __init__(self, bbox_min, bbox_max):
        self.bbox_min = np.asarray(bbox_min)
        self.bbox_max = np.asarray(bbox_max)

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC', 'TARGET_NORMAL', 'INPUT_COLOR' and 'GT_GEOMETRY'
        assert DS.INPUT_PC in data.keys()
        assert DS.GT_DENSE_PC not in data.keys()
        assert DS.GT_DENSE_NORMAL not in data.keys()

        input_pts = data[DS.INPUT_PC]
        crop_inds = np.logical_and.reduce([
            input_pts[:, 0] > self.bbox_min[0], input_pts[:, 0] < self.bbox_max[0],
            input_pts[:, 1] > self.bbox_min[1], input_pts[:, 1] < self.bbox_max[1],
            input_pts[:, 2] > self.bbox_min[2], input_pts[:, 2] < self.bbox_max[2],
        ])

        data_out = data.copy()
        data_out[DS.INPUT_PC] = input_pts[crop_inds, :]
        if DS.TARGET_NORMAL in data.keys():
            data_out[DS.TARGET_NORMAL] = data[DS.TARGET_NORMAL][crop_inds, :]

        if DS.INPUT_SENSOR_POS in data.keys():
            data_out[DS.INPUT_SENSOR_POS] = data[DS.INPUT_SENSOR_POS][crop_inds, :]

        if DS.INPUT_COLOR in data.keys():
            data_out[DS.INPUT_COLOR] = data[DS.INPUT_COLOR][crop_inds, :]

        if DS.GT_GEOMETRY in data.keys():
            data_out[DS.GT_GEOMETRY] = data_out[DS.GT_GEOMETRY].crop([
                [self.bbox_min, self.bbox_max]
            ])[0]

        return data_out


class FixedScale:
    """
    Just do a simple scaling
    """
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC', 'GT_DENSE_PC', 'GT_ONET_SAMPLE', 'GT_GEOMETRY'
        data_out = data.copy()
        assert DS.INPUT_PC in data.keys()

        points = data[DS.INPUT_PC]
        data_out[DS.INPUT_PC] = points * self.scale
        if DS.GT_DENSE_PC in data.keys():
            data_out[DS.GT_DENSE_PC] = data[DS.GT_DENSE_PC] * self.scale

        if DS.GT_ONET_SAMPLE in data.keys():
            data_out[DS.GT_ONET_SAMPLE][0] = data[DS.GT_ONET_SAMPLE][0] * self.scale

        if DS.INPUT_SENSOR_POS in data.keys():
            data_out[DS.INPUT_SENSOR_POS] = data[DS.INPUT_SENSOR_POS] * self.scale

        if DS.GT_GEOMETRY in data.keys():
            data_out[DS.GT_GEOMETRY].transform(Isometry(), self.scale)

        return data_out


class BoundScale:
    """
    Centralize the point cloud and limit the bound to [-a,a], where min_a <= a <= max_a.
    """
    def __init__(self, min_a, max_a):
        assert min_a <= max_a
        self.min_a = min_a
        self.max_a = max_a

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC', 'GT_DENSE_PC', 'GT_ONET_SAMPLE', 'GT_GEOMETRY'
        data_out = data.copy()
        assert DS.INPUT_PC in data.keys()

        points = data[DS.INPUT_PC]
        p_max, p_min = np.max(points, axis=0), np.min(points, axis=0)
        center = (p_max + p_min) / 2.
        cur_scale = np.max(p_max - p_min) / 2.
        target_scale = max(min(cur_scale, self.max_a), self.min_a)

        data_out[DS.INPUT_PC] = (points - center[None, :]) * (target_scale / cur_scale)
        if DS.GT_DENSE_PC in data.keys():
            data_out[DS.GT_DENSE_PC] = (data[DS.GT_DENSE_PC] - center[None, :]) * (target_scale / cur_scale)

        if DS.INPUT_SENSOR_POS in data.keys():
            data_out[DS.INPUT_SENSOR_POS] = (data[DS.INPUT_SENSOR_POS] - center[None, :]) * (target_scale / cur_scale)

        if DS.GT_ONET_SAMPLE in data.keys():
            data_out[DS.GT_ONET_SAMPLE][0] = (data[DS.GT_ONET_SAMPLE][0] - center[None, :]) * \
                                             (target_scale / cur_scale)

        if DS.GT_GEOMETRY in data.keys():
            data_out[DS.GT_GEOMETRY].transform(Isometry(t=-center), target_scale / cur_scale)

        return data_out


class UniformDensity:
    @classmethod
    def _compute_density(cls, xyz: np.ndarray, voxel_size: float):
        if xyz.shape[0] > 5000000:
            logger.info(f"numpy computing density for {xyz.shape[0]} points.")
        q_xyz = np.unique(np.floor_divide(xyz, voxel_size).astype(int), axis=0)
        density = xyz.shape[0] / q_xyz.shape[0]
        return density


class UniformDensityFixedSampleScale(UniformDensity):
    """
    With input points unchanged, pick a scale to reach uniform density.
        --> Used when you want to use all input information.
    """
    def __init__(self, voxel_size, min_density, max_density):
        self.voxel_size = voxel_size
        self.min_density = min_density
        self.max_density = max_density
        assert self.min_density <= self.max_density

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC', 'GT_DENSE_PC', 'GT_ONET_SAMPLE', 'GT_GEOMETRY'
        data_out = data.copy()
        assert DS.INPUT_PC in data.keys()

        points = data[DS.INPUT_PC]
        cur_density = self._compute_density(points, self.voxel_size)
        target_density = rng.uniform(self.min_density, self.max_density)

        # If you want to increase density to 4x, then you must down-scale pc by 1/2.
        target_scale = np.sqrt(cur_density / target_density)

        data_out[DS.INPUT_PC] = points * target_scale
        if DS.GT_DENSE_PC in data.keys():
            data_out[DS.GT_DENSE_PC] = data[DS.GT_DENSE_PC] * target_scale

        if DS.GT_ONET_SAMPLE in data.keys():
            data_out[DS.GT_ONET_SAMPLE][0] = data[DS.GT_ONET_SAMPLE][0] * target_scale

        if DS.INPUT_SENSOR_POS in data.keys():
            data_out[DS.INPUT_SENSOR_POS] = data[DS.INPUT_SENSOR_POS] * target_scale

        if DS.GT_GEOMETRY in data.keys():
            data_out[DS.GT_GEOMETRY].transform(scale=target_scale)

        return data_out


class UniformDensityFixedScaleSample(UniformDensity):
    """
    With scale unchanged, randomly sub-sample points to reach uniform density.
        --> This is not poisson disk! Internal variation will still be kept.
            The desired density is in the average sense.
        --> Used when input point is synthetically dense and can be sub-sampled,
            and it's easier to control the scale
    """
    def __init__(self, voxel_size, min_density, max_density):
        self.voxel_size = voxel_size
        self.min_density = min_density
        self.max_density = max_density
        assert self.min_density <= self.max_density

    def __call__(self, data, rng):
        data_out = data.copy()
        assert DS.INPUT_PC in data.keys()

        points = data[DS.INPUT_PC]
        cur_density = self._compute_density(points, self.voxel_size)

        if cur_density < self.min_density:
            logger.warning(f"UniformDensity - FixedScaleSample: Cannot subsample when current density is only "
                           f"{cur_density}, desired min = {self.min_density}")

        target_density = rng.uniform(min(cur_density, self.min_density), min(cur_density, self.max_density))
        target_n = np.round(points.shape[0] / cur_density * target_density).astype(int).item()

        indices = pad_cloud(points, target_n, return_inds=True, random_state=rng)
        data_out[DS.INPUT_PC] = points[indices, :]

        if DS.TARGET_NORMAL in data.keys():
            data_out[DS.TARGET_NORMAL] = data[DS.TARGET_NORMAL][indices, :]

        if DS.INPUT_SENSOR_POS in data.keys():
            data_out[DS.INPUT_SENSOR_POS] = data[DS.INPUT_SENSOR_POS][indices, :]

        if DS.INPUT_COLOR in data.keys():
            data_out[DS.INPUT_COLOR] = data[DS.INPUT_COLOR][indices, :]

        return data_out


class Centralize:
    """
    Centralize the point cloud only without BoundScale, with optional noise added to the final center
    """
    def __init__(self, noise: float = 0.0):
        self.noise = noise

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC', 'GT_DENSE_PC', 'GT_ONET_SAMPLE', 'GT_GEOMETRY'
        data_out = data.copy()
        assert DS.INPUT_PC in data.keys()

        points = data[DS.INPUT_PC]
        p_max, p_min = np.max(points, axis=0), np.min(points, axis=0)
        center = (p_max + p_min) / 2.

        center_noise = (p_max - p_min) * rng.uniform([-self.noise] * 3, [self.noise] * 3)
        center += center_noise

        data_out[DS.INPUT_PC] = points - center[None, :]
        if DS.GT_DENSE_PC in data.keys():
            data_out[DS.GT_DENSE_PC] = data[DS.GT_DENSE_PC] - center[None, :]

        if DS.GT_ONET_SAMPLE in data.keys():
            data_out[DS.GT_ONET_SAMPLE][0] = data[DS.GT_ONET_SAMPLE][0] - center[None, :]

        if DS.INPUT_SENSOR_POS in data.keys():
            data_out[DS.INPUT_SENSOR_POS] = data[DS.INPUT_SENSOR_POS] - center[None, :]

        if DS.GT_GEOMETRY in data.keys():
            data_out[DS.GT_GEOMETRY].transform(Isometry(t=-center), 1.0)

        return data_out


class FixedAxisRotation:
    """
    (randomly) rotate the point cloud, with fixed axis and degrees in a certain range.
    """
    def __init__(self, axis, deg_min, deg_max):
        if isinstance(axis, str):
            axis = Isometry._str_to_axis(axis)
        self.axis = np.asarray(axis)
        self.deg_min = deg_min
        self.deg_max = deg_max

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC', 'TARGET_NORMAL', 'GT_DENSE_PC', 'GT_DENSE_NORMAL', 'GT_ONET_SAMPLE'
        data_out = data.copy()
        assert DS.GT_GEOMETRY not in data.keys()

        rot_degree = rng.uniform(self.deg_min, self.deg_max)
        rot_iso = Isometry.from_axis_angle(self.axis, degrees=rot_degree)

        data_out[DS.INPUT_PC] = rot_iso @ data[DS.INPUT_PC]

        if DS.TARGET_NORMAL in data.keys():
            data_out[DS.TARGET_NORMAL] = rot_iso @ data[DS.TARGET_NORMAL]

        if DS.GT_DENSE_PC in data.keys():
            data_out[DS.GT_DENSE_PC] = rot_iso @ data[DS.GT_DENSE_PC]

        if DS.GT_DENSE_NORMAL in data.keys():
            data_out[DS.GT_DENSE_NORMAL] = rot_iso @ data[DS.GT_DENSE_NORMAL]

        if DS.GT_ONET_SAMPLE in data.keys():
            data_out[DS.GT_ONET_SAMPLE][0] = rot_iso @ data[DS.GT_ONET_SAMPLE][0]

        if DS.INPUT_SENSOR_POS in data.keys():
            data_out[DS.INPUT_SENSOR_POS] = rot_iso @ data[DS.INPUT_SENSOR_POS]

        return data_out


class ComposedTransforms:
    def __init__(self, args):
        self.args = args
        self.transforms = []
        if self.args is not None:
            for t_spec in self.args:
                self.transforms.append(
                    globals()[t_spec.name](**t_spec.args)
                )

    def __call__(self, data, rng):
        for t in self.transforms:
            data = t(data, rng)
        return data
