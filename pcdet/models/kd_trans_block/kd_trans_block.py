import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.spconv_utils import spconv
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ...utils import common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms, class_agnostic_nms_minor
from ...ops.iou3d_nms import iou3d_nms_utils


class KDPointTrans(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    # @torch.no_grad()
    def proposal_layer_stu2tea(self, batch_dict, nms_config):

        batch_size = batch_dict['batch_size']
        batch_box_preds_tea = batch_dict['batch_box_preds_tea_densehead']
        batch_cls_preds_tea = batch_dict['batch_cls_preds_tea_densehead']

        batch_box_preds_stu = batch_dict['batch_box_preds']
        batch_cls_preds_stu = batch_dict['batch_cls_preds']

        dense_cls_preds_stu = batch_dict['cls_preds'].view(batch_size, -1, batch_cls_preds_stu.shape[-1])
        if self.model_cfg.get('USE_REG', None) and self.model_cfg.USE_REG:
            dense_reg_preds_stu = batch_dict['box_preds'].view(batch_size, -1, batch_box_preds_stu.shape[-1])

        ''' stu-to-tea'''
        cls_select_reverse = dense_cls_preds_stu.new_zeros(
            (batch_size, nms_config.NMS_POST_MAXSIZE, batch_cls_preds_stu.shape[-1]))
        rois_reverse = batch_box_preds_tea.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE,
                                                      batch_box_preds_tea.shape[-1]))

        if self.model_cfg.get('USE_REG', None) and self.model_cfg.USE_REG:
            box_preds_stu_ori = dense_reg_preds_stu.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE,
                                                      batch_box_preds_tea.shape[-1]))

        select_mask = batch_box_preds_tea.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds_tea.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_box_preds_tea[batch_mask]
            box_preds_stu = batch_box_preds_stu[batch_mask]
            cls_preds_stu = batch_cls_preds_stu[batch_mask]
            cur_roi_scores_stu, cur_roi_labels_stu = torch.max(cls_preds_stu, dim=1)

            if self.model_cfg.get('USE_REG', None) and self.model_cfg.USE_REG:
                box_preds_stu_ori_tmp = dense_reg_preds_stu[batch_mask]

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                # TODO: add for "student ---> select teacher"
                selected_minor, selected_scores_minor = class_agnostic_nms_minor(
                    box_scores=cur_roi_scores_stu, box_preds=box_preds_stu, nms_config=nms_config,
                    score_thresh_minor=nms_config.SCORE_THRESH_MINOR,
                )

            ''' stu-to-tea '''
            cur_cls_preds = dense_cls_preds_stu[batch_mask]
            rois_reverse[index, :len(selected_minor), :] = box_preds[selected_minor]
            cls_select_reverse[index, :len(selected_minor), :] = cur_cls_preds[selected_minor]
            select_mask[index, :len(selected_minor)] = 1.0
            if self.model_cfg.get('USE_REG', None) and self.model_cfg.USE_REG:
                box_preds_stu_ori[index, :len(selected_minor), :] = box_preds_stu_ori_tmp[selected_minor]

        # TODO: ''' stu-to-tea '''
        batch_dict['rois_new_tea'] = rois_reverse
        batch_dict['cls_preds_selected_kd'] = cls_select_reverse
        batch_dict['select_mask'] = select_mask
        if self.model_cfg.get('USE_REG', None) and self.model_cfg.USE_REG:
            batch_dict['box_preds_stu_ori'] = box_preds_stu_ori

        return batch_dict


    def forward(self, batch_dict):

        # if self.model_cfg.NMS_CONFIG.MODE == 'Stu2Tea':
        _ = self.proposal_layer_stu2tea(batch_dict, nms_config=self.model_cfg.NMS_CONFIG)

        batch_dict['re_run_tea_flag'] = True
        roi_tmp = batch_dict['rois'].detach()
        batch_dict['rois'] = batch_dict['rois_new_tea'].detach()
        teacher_model = batch_dict['teacher_model']
        teacher_model.roi_head(batch_dict)
        batch_dict['rois'] = roi_tmp.detach()
        batch_dict['re_run_tea_flag'] = None

        select_obj_mask = batch_dict['select_mask'].view(-1, 1)

        batch_dict['cls_preds_selected_kd'] = batch_dict['cls_preds_selected_kd'].view(-1, batch_dict['cls_preds_selected_kd'].shape[-1])
        cls_pred_selected_reverse_max, _ = batch_dict['cls_preds_selected_kd'].max(dim=1, keepdim=True)

        batch_dict['cls_preds_kd_stu'] = cls_pred_selected_reverse_max * select_obj_mask
        batch_dict['cls_preds_kd_tea'] = batch_dict['rcnn_cls_stu_to_tea']* select_obj_mask



        # if self.model_cfg.get('USE_REG', None) and self.model_cfg.USE_REG:
        #     batch_dict['box_preds_kd_stu'] = batch_dict['box_preds_stu_ori'].view(-1, batch_dict['rois_new_tea'].shape[-1]) * select_obj_mask
        #     batch_dict['box_preds_kd_tea'] = batch_dict['rcnn_reg_stu_to_tea'] * select_obj_mask


        return batch_dict
