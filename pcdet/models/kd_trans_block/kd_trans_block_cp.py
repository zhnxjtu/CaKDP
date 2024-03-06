import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ...utils.spconv_utils import spconv
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ...utils import common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms, class_agnostic_nms_minor
from .target_assigned_teacher.proposal_taget_layer_teacher import ProposalTargetLayer
from ..model_utils import centernet_utils


class KDPointTrans_cp(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, K=40):
        batch, num_class, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_classes = (topk_ind // K).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_classes, topk_ys, topk_xs

    def decode_bbox_from_heatmap(self, heatmap, rot_cos, rot_sin, center, center_z, dim,
                                 point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, K=100,
                                 circle_nms=False, score_thresh=None, post_center_limit_range=None):
        batch_size, num_class, _, _ = heatmap.size()

        # if circle_nms:
        #     # TODO: not checked yet
        #     assert False, 'not checked yet'
        #     heatmap = _nms(heatmap)

        scores, inds, class_ids, ys, xs = self._topk(heatmap, K=K)
        center = self._transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
        rot_sin = self._transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
        rot_cos = self._transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
        center_z = self._transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
        dim = self._transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)

        angle = torch.atan2(rot_sin, rot_cos)
        xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

        xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
        ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]
        if vel is not None:
            vel = self._transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
            box_part_list.append(vel)

        final_box_preds = torch.cat((box_part_list), dim=-1)
        final_scores = scores.view(batch_size, K)
        final_class_ids = class_ids.view(batch_size, K)

        assert post_center_limit_range is not None
        mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

        if score_thresh is not None:
            mask &= (final_scores > score_thresh)

        ret_pred_dicts = []
        for k in range(batch_size):
            cur_mask = mask[k]
            cur_boxes = final_box_preds[k, cur_mask]
            cur_scores = final_scores[k, cur_mask]
            cur_labels = final_class_ids[k, cur_mask]
            cur_index = inds[k, cur_mask]

            ret_pred_dicts.append({
                'pred_boxes': cur_boxes,
                'pred_scores': cur_scores,
                'pred_labels': cur_labels,
                'index': cur_index,
            })
        return ret_pred_dicts

    def proposal_layer(self, batch_dict, nms_config):
        post_process_cfg = nms_config.POST_CONFIG
        post_center_limit_range = torch.tensor(self.point_cloud_range).cuda().float()

        pred_dicts = batch_dict['pred_dicts_stu']
        pred_dicts_before = batch_dict['pred_dicts_stu']
        batch_size = batch_dict['batch_size']

        # feature_number = batch_dict['batch_box_preds_tea_densehead'].shape[-1]
        batch_box_preds_tea = batch_dict['batch_box_preds_tea_densehead'].view(batch_size, 200, 176, 6, -1)
        batch_cls_preds_tea = torch.sigmoid(batch_dict['batch_cls_preds_tea_densehead'].view(batch_size, 200, 176, 6, -1))
        batch_cls_preds_tea_index = batch_cls_preds_tea.max(-1)[0].max(-1,keepdim=True)[1].unsqueeze(-1).repeat(1,1,1,1,7)
        batch_box_preds_tea = batch_box_preds_tea.gather(dim=3, index = batch_cls_preds_tea_index).squeeze()


        # ret_dict = [{
        #     'pred_boxes': [],
        #     'pred_scores': [],
        #     'pred_labels': [],
        # } for k in range(batch_size)]

        for idx, pred_dict in enumerate(pred_dicts):

            if idx > 0:
                assert False

            batch_hm = pred_dict['hm'].sigmoid()
            # print(batch_box_preds_tea.shape)
            # exit()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in post_process_cfg.HEAD_ORDER else None

            final_pred_dicts = self.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=post_process_cfg.FEATURE_MAP_STRIDE,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )


            rois_reverse = batch_box_preds_tea.new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, 7))
            cls_select_reverse = batch_box_preds_tea.new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, pred_dict['hm'].shape[1]))
            mask = batch_box_preds_tea.new_zeros((batch_size, nms_config.TRAIN.NMS_POST_MAXSIZE, 1))
            for k, final_dict in enumerate(final_pred_dicts):
                # final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                # print(final_dict['pred_scores'].sort())
                # exit()
                selected, selected_scores = class_agnostic_nms_minor(
                    box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                    nms_config=nms_config.TRAIN,
                    score_thresh_minor=nms_config.TRAIN.SCORE_THRESH_MINOR,
                )

                # final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                # final_dict['pred_scores'] = selected_scores
                # final_dict['index'] = final_dict['index'][selected]
                # final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                # ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                # ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                # ret_dict[k]['index'].append(final_dict['index'])
                # ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])


                cur_box_preds_tea = batch_box_preds_tea[k]
                # cur_cls_preds_tea = batch_cls_preds_tea[k]
                select_index = final_dict['index'][selected]


                # print(cur_box_preds_tea.shape)
                with torch.no_grad():
                    rois_reverse[k, :len(selected), :] = cur_box_preds_tea[select_index//176, select_index%176, :]
                    mask[k, :len(selected), :] = 1.0
                # exit()
                cls_select_reverse[k, :len(selected), :] = pred_dicts_before[idx]['hm'][k, :, select_index//176, select_index%176].t()


                # cur_box_preds_tea_select = cur_box_preds_tea[select_index//176, select_index%176, :]
                # roi_tea_list.append(cur_box_preds_tea_select)

        # for k in range(batch_size):
        #     ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
        #     ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
        #     # ret_dict[k]['index'] = torch.cat(ret_dict[k]['index'], dim=0)
        #     # ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        batch_dict['rois_reverse'] = rois_reverse
        batch_dict['cls_preds_selected_reverse'] = cls_select_reverse
        batch_dict['mask_use_item'] = mask

        return batch_dict

    def forward(self, batch_dict):

        _ = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG
        )

        with torch.no_grad():
            batch_dict['re_run_tea_flag'] = True
            # roi_tmp = batch_dict['rois'].detach()
            batch_dict['rois'] = batch_dict['rois_reverse'].detach().view(batch_dict['batch_size'], -1, batch_dict['batch_box_preds_tea_densehead'].shape[-1])
            # teacher_model = batch_dict['teacher_model']
            batch_dict['teacher_model'].roi_head(batch_dict)
            # batch_dict['rois'] = roi_tmp.detach()
            batch_dict['re_run_tea_flag'] = None
            select_obj_mask = batch_dict['mask_use_item'].view(-1, 1)
            # cnn_cls_stu_to_tea = batch_dict['rcnn_cls_stu_to_tea'].view(batch_dict['batch_size'], -1, self.model_cfg.ANCHOR_NUM_PER_LOCATION)
            # cnn_cls_stu_to_tea = cnn_cls_stu_to_tea.max(-1)[0].view(-1, 1)
            batch_dict['cls_preds_selected_reverse_tea'] = batch_dict['rcnn_cls_stu_to_tea'] * select_obj_mask



        # batch_dict['cls_preds_selected_reverse'] =\
        #     batch_dict['cls_preds_selected_reverse'].view(-1, batch_dict['cls_preds_selected_reverse'].shape[-1])
        # select_obj_mask = (batch_dict['cls_preds_selected_reverse'].sum(-1) != 0).unsqueeze(-1)

        cls_pred_selected_reverse_max = batch_dict['cls_preds_selected_reverse'].max(dim=-1)[0].view(-1, 1)
        batch_dict['cls_preds_selected_reverse_stu'] = cls_pred_selected_reverse_max * select_obj_mask


        return batch_dict
