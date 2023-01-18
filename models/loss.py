import ext

import torch
from nksr.svh import SparseFeatureHierarchy
from dataset.base import DatasetSpec as DS
import torch.nn.functional as F


class KitchenSinkMetricLoss:
    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        raise NotImplementedError

    @classmethod
    def _get_svh_samples(cls, svh: SparseFeatureHierarchy,
                         n_samples: int, expand: int = 0, expand_top: int = 0):
        """
        Get random samples, across all layers of the decoder hierarchy
        :param n_samples: int, number of total samples
        :param expand: size of expansion
        :param expand_top: size of expansion of the coarsest level.
        :return: (n_samples, 3)
        """
        base_coords, base_scales = [], []
        for d in range(svh.depth):
            if svh.vdbs[d] is None:
                continue
            ijk_coords = svh.vdbs[d].active_grid_coords()
            d_expand = expand if d != svh.depth - 1 else expand_top
            if d_expand >= 3:
                mc_offsets = torch.arange(-d_expand // 2 + 1, d_expand // 2 + 1, device=svh.device)
                mc_offsets = torch.stack(torch.meshgrid(mc_offsets, mc_offsets, mc_offsets, indexing='ij'), dim=3)
                mc_offsets = mc_offsets.view(-1, 3)
                ijk_coords = (ijk_coords.unsqueeze(dim=1).repeat(1, mc_offsets.size(0), 1) +
                              mc_offsets.unsqueeze(0)).view(-1, 3)
                ijk_coords = torch.unique(ijk_coords, dim=0)
            base_coords.append(svh.vdbs[d].grid_to_world(ijk_coords.float()))
            base_scales.append(torch.full((ijk_coords.size(0), ), svh.vdbs[d].voxel_size, device=svh.device))
        base_coords, base_scales = torch.cat(base_coords), torch.cat(base_scales)
        local_ids = (torch.rand((n_samples, ), device=svh.device) * base_coords.size(0)).long()
        local_coords = (torch.rand((n_samples, 3), device=svh.device) - 0.5) * base_scales[local_ids, None]
        query_pos = base_coords[local_ids] + local_coords
        return query_pos

    @classmethod
    def _get_samples(cls, hparams, configs, svh, ref_xyz, ref_normal):
        all_samples = []
        for config in configs:
            if config.type == "uniform":
                all_samples.append(
                    cls._get_svh_samples(svh, config.n_samples, config.expand, config.expand_top)
                )
            elif config.type == "band":
                band_inds = (torch.rand((config.n_samples, ), device=ref_xyz.device) * ref_xyz.size(0)).long()
                eps = config.eps * hparams.voxel_size
                band_pos = ref_xyz[band_inds] + \
                    ref_normal[band_inds] * torch.randn((config.n_samples, 1), device=ref_xyz.device) * eps
                all_samples.append(band_pos)
        return torch.cat(all_samples, 0)

    @classmethod
    def transform_field(cls, hparams, field: torch.Tensor):
        spatial_config = hparams.supervision.spatial
        assert spatial_config.gt_type != "binary"
        truncation_size = spatial_config.gt_band * hparams.voxel_size
        # non-binary supervision (made sure derivative norm at 0 if 1)
        if spatial_config.gt_soft:
            field = torch.tanh(field / truncation_size) * truncation_size
        else:
            field = torch.clone(field)
            field[field > truncation_size] = truncation_size
            field[field < -truncation_size] = -truncation_size
        return field

    @classmethod
    def compute_gt_chi_from_pts(cls, hparams, query_pos: torch.Tensor, ref_xyz: torch.Tensor, ref_normal: torch.Tensor):
        mc_query_sdf = -ext.sdfgen.sdf_from_points(query_pos, ref_xyz, ref_normal, 8, 0.02, False)[0]
        return cls.transform_field(hparams, mc_query_sdf)


class ShapeNetIoUMetric(KitchenSinkMetricLoss):
    """
    Will only output for ShapeNet data.
    """
    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        if compute_metric:
            if DS.GT_ONET_SAMPLE not in batch.keys():
                return
            with torch.no_grad():
                iou_pd = out['field'].evaluate_f_bar(batch[DS.GT_ONET_SAMPLE][0][0]) > 0
                iou_gt = batch[DS.GT_ONET_SAMPLE][1][0] > 0
            iou = torch.sum(torch.logical_and(iou_pd, iou_gt)) / (
                    torch.sum(torch.logical_or(iou_pd, iou_gt)) + 1.0e-6)
            metric_dict.add_loss('iou', iou)


class UDFLoss(KitchenSinkMetricLoss):
    """
    UDF Loss for supervising the UDF branch
    """
    @classmethod
    def compute_gt_tudf(cls, chi_pos, hparams, ref_xyz, ref_normal, ref_geometry):
        if ref_geometry is not None:
            gt_tsdf = cls.transform_field(hparams, ref_geometry.query_sdf(chi_pos))
        else:
            gt_tsdf = cls.compute_gt_chi_from_pts(hparams, chi_pos, ref_xyz, ref_normal)
        gt_tudf = torch.abs(gt_tsdf)
        return gt_tudf

    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        udf_config = hparams.supervision.udf
        field = out['field']
        if hparams.udf.enabled and udf_config.weight > 0.0:

            if DS.GT_GEOMETRY not in batch.keys():
                ref_geometry = None
                ref_xyz, ref_normal = batch[DS.GT_DENSE_PC][0], batch[DS.GT_DENSE_NORMAL][0]
            else:
                ref_geometry = batch[DS.GT_GEOMETRY][0]
                ref_xyz, ref_normal, _ = ref_geometry.torch_attr()

            udf_field = field.mask_field
            chi_pos = cls._get_samples(hparams, udf_config.samplers, field.svh, ref_xyz, ref_normal)
            pd_chi = udf_field.evaluate_f(chi_pos).value

            gt_tudf = cls.compute_gt_tudf(chi_pos, hparams, ref_xyz, ref_normal, ref_geometry)
            pd_tudf = cls.transform_field(hparams, pd_chi)
            udf_loss_normalized = torch.mean(torch.abs(pd_tudf - gt_tudf) / hparams.voxel_size)

            loss_dict.add_loss(f"udf", udf_loss_normalized, udf_config.weight)


class StructureLoss(KitchenSinkMetricLoss):
    """
    Cross entropy of the voxel classification
    (will also output accuracy metric)
    """
    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        if hparams.supervision.structure_weight > 0.0:
            gt_svh = out['gt_svh']
            for feat_depth, struct_feat in out['structure_features'].items():
                if struct_feat.size(0) == 0:
                    continue
                gt_status = gt_svh.evaluate_voxel_status(out['dec_tmp_svh'].vdbs[feat_depth], feat_depth)
                loss_dict.add_loss(f"struct-{feat_depth}", F.cross_entropy(struct_feat, gt_status),
                                   hparams.supervision.structure_weight)
                if compute_metric:
                    metric_dict.add_loss(f"struct-acc-{feat_depth}",
                                         torch.mean((struct_feat.argmax(dim=1) == gt_status).float()))


class GTSurfaceLoss(KitchenSinkMetricLoss):
    """
    1. L1 Loss on the surface
    2. Dot-product loss on the surface normals
    """
    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        gt_surface_config = hparams.supervision.gt_surface
        field = out['field']

        if gt_surface_config.value > 0.0 or gt_surface_config.normal > 0.0:

            if DS.GT_GEOMETRY not in batch.keys():
                ref_xyz, ref_normal = batch[DS.GT_DENSE_PC][0], batch[DS.GT_DENSE_NORMAL][0]
            else:
                ref_geometry = batch[DS.GT_GEOMETRY][0]
                ref_xyz, ref_normal, _ = ref_geometry.torch_attr()

            n_subsample = gt_surface_config.subsample
            if 0 < n_subsample < ref_xyz.size(0):
                ref_xyz_inds = (torch.rand((n_subsample,), device=ref_xyz.device) *
                                ref_xyz.size(0)).long()
            else:
                ref_xyz_inds = torch.arange(ref_xyz.size(0), device=ref_xyz.device)

            compute_grad = gt_surface_config.normal > 0.0
            eval_res = field.evaluate_f(ref_xyz[ref_xyz_inds], grad=compute_grad)

            if compute_grad:
                pd_grad = eval_res.gradient
                pd_grad = -pd_grad / (torch.linalg.norm(pd_grad, dim=-1, keepdim=True) + 1.0e-6)
                loss_dict.add_loss('gt-surface-normal',
                                   1.0 - torch.sum(pd_grad * ref_normal[ref_xyz_inds], dim=-1).mean(),
                                   gt_surface_config.normal)

            loss_dict.add_loss('gt-surface-value', torch.abs(eval_res.value).mean(), gt_surface_config.value)


class SpatialLoss(KitchenSinkMetricLoss):
    """
    1. TSDF-Loss:
        - Near Surface: L1 of TSDF
        - Far Surface: (ShapeNet does not contain this region) exp
    2. RegSDF-Loss
    """

    @classmethod
    def apply(cls, hparams, loss_dict, metric_dict, batch, out, compute_metric):
        opt = hparams.supervision.spatial
        field = out['field']

        if DS.GT_GEOMETRY not in batch.keys():
            ref_geometry = None
            ref_xyz, ref_normal = batch[DS.GT_DENSE_PC][0], batch[DS.GT_DENSE_NORMAL][0]
        else:
            ref_geometry = batch[DS.GT_GEOMETRY][0]
            ref_xyz, ref_normal, _ = ref_geometry.torch_attr()

        if opt.weight > 0.0:
            chi_pos = cls._get_samples(hparams, opt.samplers, field.svh, ref_xyz, ref_normal)

            # Note: If expand <= 3 then chi_mask will always be valid.
            pd_chi = field.evaluate_f(chi_pos).value

            if ref_geometry is not None:
                gt_sdf = ref_geometry.query_sdf(chi_pos)
                gt_tsdf = cls.transform_field(hparams, gt_sdf)

                gt_cls = ref_geometry.query_classification(chi_pos)
                near_surface_mask = gt_cls == 0
                empty_space_mask = gt_cls == 1

            else:
                gt_tsdf = cls.compute_gt_chi_from_pts(hparams, chi_pos, ref_xyz, ref_normal)

                near_surface_mask = torch.ones(chi_pos.size(0), dtype=bool, device=chi_pos.device)
                empty_space_mask = ~near_surface_mask

            pd_tsdf = cls.transform_field(hparams, pd_chi)
            near_surface_l1 = torch.abs(
                (pd_tsdf[near_surface_mask] - gt_tsdf[near_surface_mask]) / hparams.voxel_size)

            # Empty space: value as small as possible.
            empty_scale = 2.0 * hparams.voxel_size
            empty_space_loss = 0.1 * torch.exp(pd_chi[empty_space_mask] / empty_scale)
            mixed_loss = (torch.sum(near_surface_l1) + torch.sum(empty_space_loss)) / chi_pos.size(0)
            loss_dict.add_loss(f"spatial", mixed_loss, opt.weight)

            # RegSDF Loss:
            if opt.reg_sdf_weight > 0.0:
                reg_sdf_eps = 0.5
                reg_sdf_loss = torch.mean(reg_sdf_eps / (pd_chi ** 2 + reg_sdf_eps ** 2))
                loss_dict.add_loss(f"msa", reg_sdf_loss, opt.reg_sdf_weight)
