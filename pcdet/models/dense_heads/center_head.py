import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils, common_utils
# KD modules
from pcdet.utils import box_utils
from pcdet.models.model_utils.efficientnet_utils import get_act_layer
from pcdet.models.model_utils.batch_norm_utils import get_norm_layer
from pcdet.models.model_utils.basic_block_2d import build_block


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False, act_fn=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, upsample=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                if not upsample:
                    block = [nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)]
                else:
                    block = [nn.ConvTranspose2d(input_channels, input_channels, 2, stride=2, bias=use_bias)]

                block.extend([norm_layer(input_channels), act_fn()])
                fc_list.append(nn.Sequential(*block))

            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        act_fn = get_act_layer(self.model_cfg.get('ACT_FN', 'ReLU'))
        norm_layer = get_norm_layer(self.model_cfg.get('NORM_TYPE', 'BatchNorm2d'))

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        # build pre block
        if self.model_cfg.get('PRE_BLOCK', None):
            pre_block = []

            block_types = self.model_cfg.PRE_BLOCK.BLOCK_TYPE
            num_filters = self.model_cfg.PRE_BLOCK.NUM_FILTERS
            layer_strides = self.model_cfg.PRE_BLOCK.LAYER_STRIDES
            kernel_sizes = self.model_cfg.PRE_BLOCK.KERNEL_SIZES
            paddings = self.model_cfg.PRE_BLOCK.PADDINGS
            in_channels = input_channels
            
            for i in range(len(num_filters)):
                pre_block.extend(build_block(
                    block_types[i], in_channels, num_filters[i], kernel_size=kernel_sizes[i],
                    stride=layer_strides[i], padding=paddings[i], bias=False
                ))
                in_channels = num_filters[i]
            self.pre_block = nn.Sequential(*pre_block)

        # build shared block
        if self.model_cfg.get('SHARED_CONV_UPSAMPLE', None):
            shared_block = [nn.ConvTranspose2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 2, stride=2, padding=0,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            )]
        elif self.model_cfg.get('SHARED_CONV_DOWNSAMPLE', None):
            shared_block = [nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=2, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            )]
        else:
            shared_block = [nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            )]
        shared_block.extend([norm_layer(self.model_cfg.SHARED_CONV_CHANNEL), act_fn()])
        self.shared_conv = nn.Sequential(*shared_block)

        # build separate head
        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            if self.model_cfg.get('NUM_IOU_CONV', None):
                cur_head_dict['iou'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_IOU_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    act_fn=act_fn,
                    norm_layer=norm_layer,
                    upsample=self.separate_head_cfg.get('UPSAMPLE', False)
                )
            )
        self.predict_boxes_when_training = self.model_cfg.get('PRED_BOX_WHEN_TRAIN', predict_boxes_when_training)

        self.forward_ret_dict = {}

        # for kd only
        self.is_teacher = False
        self.kd_head = None
        self.disable = self.model_cfg.get('DISABLE', None)
        self.disable_inference = self.model_cfg.get('DISABLE_INFERENCE', None)

        self.build_losses()

        # for inference time analysis only
        # self.time_meter = common_utils.DictAverageMeter()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())
        self.add_module('iou_loss_func', loss_utils.IOULossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2, hm_filter_num=None
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        iou = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])  # [3,200,176]

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()

            if hm_filter_num is not None and k < hm_filter_num:
                pass
            else:
                centernet_utils.draw_gaussian_to_heatmap(
                    heatmap[cur_class_id], center[k], radius[k].item(),
                    sharper=self.model_cfg.TARGET_ASSIGNER_CONFIG.get('SHARPER', None)
                )

            # voxel index
            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

            iou[cur_class_id, center_int[k, 1], center_int[k, 0]] = 1

        return heatmap, ret_boxes, inds, mask, iou

    def assign_targets(self, gt_boxes, feature_map_size=None, hm_filter_num_list=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            feature_map_size: (2) [H, W]
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'ious': [],
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, iou_list = [], [], [], [], []

            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask, iou = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                    hm_filter_num=None if hm_filter_num_list is None else hm_filter_num_list[bs_idx]
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                iou_list.append(iou.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['ious'].append(torch.stack(iou_list, dim=0))

        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            if self.model_cfg.get('NUM_IOU_CONV', None):
                # # TODO: iou loss ####
                # pred_dict['iou'] = self.sigmoid(pred_dict['iou'])
                # bs, cls_num, _, _ = target_dicts['ious'][idx].shape
                # iou_weight = (target_dicts['ious'][idx].view(bs, cls_num, -1) >= 0)
                # positive_iou = iou_weight.sum(-1, keepdim=True).float()
                # iou_weight= iou_weight / torch.clamp(positive_iou, min=1.0)
                # iou_loss_src = self.iou_loss_func(pred_dict['iou'], target_dicts['ious'][idx], iou_weight)
                # iou_loss = iou_loss_src.sum() / bs
                # iou_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']
                # # print(iou_loss)

                # TODO: iou loss ####
                pred_dict['iou'] = self.sigmoid(pred_dict['iou'])
                bs, cls_num, _, _ = target_dicts['ious'][idx].shape
                target_iou_tmp = target_dicts['ious'][idx].view(bs, cls_num, -1)
                iou_weight = (target_iou_tmp >= 0)  # positive sample
                positive_iou = (target_iou_tmp > 0).sum(-1, keepdim=True).float()
                # pos_num = (target_iou_tmp.max(1)[0] > 0.0).sum(-1, keepdim=True)
                # if self.model_cfg.IOU_THRESHOLD.LOW_THRE == 0:
                #     neg_index = (target_iou_tmp.max(1)[0] == self.model_cfg.IOU_THRESHOLD.LOW_THRE)
                # else:
                #     neg_index = (target_iou_tmp.max(1)[0] <= self.model_cfg.IOU_THRESHOLD.LOW_THRE)
                # box_iou_preds_measure = pred_dict['iou'].clone().detach().view(bs, cls_num, -1).max(1)[0]
                # box_iou_preds_measure = (box_iou_preds_measure - target_iou_tmp.max(1)[0]).abs() * neg_index
                # for b_c in range(bs):
                #     max_num = min(self.model_cfg.IOU_THRESHOLD.TIMES_NEG * np.int(pos_num[b_c]), neg_index[b_c].sum())  # 2 500; 3 1000
                #     # max_num = neg_index[b_c].sum()
                #     top_value, top_index = torch.topk(box_iou_preds_measure[b_c], max_num)
                #     iou_weight[b_c, :, top_index] = 1.0

                iou_weight = iou_weight / torch.clamp(positive_iou, min=1.0)
                iou_loss_src = self.iou_loss_func(pred_dict['iou'], target_dicts['ious'][idx], iou_weight)
                iou_loss = iou_loss_src.sum() / bs
                iou_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']
                # print(iou_loss)
            #######################

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss + iou_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
            tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts, no_nms=False, nms_config=None):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            if self.model_cfg.get('NUM_IOU_CONV', None) and not self.training:
                batch_iou = pred_dict['iou'].sigmoid()
                batch_hm = batch_hm * (batch_iou > post_process_cfg.POST_IOU_THRESH).float()

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms' and not no_nms:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG if nms_config is None else nms_config,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        if self.disable or (not self.training and self.disable_inference):
            return data_dict

        infeature_name = data_dict.pop('dense_head_infeature_name', 'spatial_features_2d')
        spatial_features_2d = data_dict[infeature_name]

        in_feature_2d = spatial_features_2d
        if hasattr(self, 'pre_block'):
            in_feature_2d = self.pre_block(in_feature_2d)
            data_dict['spatial_features_2d_preblock'] = in_feature_2d

        shared_features_2d = self.shared_conv(in_feature_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(shared_features_2d))

        if self.training or self.model_cfg.get('FORCE_ASSIGN_TARGET', None):
            target_boxes = data_dict['gt_boxes']

            # kd operations
            hm_filter_num_list = None
            if self.kd_head is not None and not self.is_teacher and self.model_cfg.get('LABEL_ASSIGN_KD', None):
                target_boxes, num_target_boxes_list = self.kd_head.parse_teacher_pred_to_targets(
                    kd_cfg=self.model_cfg.LABEL_ASSIGN_KD, pred_boxes_tea=data_dict['decoded_pred_tea'],
                    gt_boxes=target_boxes
                )

            target_dict = self.assign_targets(
                target_boxes, feature_map_size=pred_dicts[0]['hm'].size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None),
                hm_filter_num_list=hm_filter_num_list
            )

            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.is_teacher and self.kd_head:
            data_dict['pred_dicts_stu'] = pred_dicts

        # add needed predictions to dict
        if self.kd_head is not None and self.is_teacher:
            self.kd_head.put_pred_to_ret_dict(self, data_dict, pred_dicts)

        if (not self.training and not self.is_teacher) or self.predict_boxes_when_training:
            # postpro_time = time.time()
            pred_dicts_tmp = pred_dicts
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if (self.predict_boxes_when_training and self.training) or (not self.training and self.model_cfg.get('PRED_BOX_WHEN_EVAL', None)):
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

                cur_cls_preds_vina = []
                for key, value in enumerate(pred_dicts_tmp):
                    cur_cls_preds_vina.append(value['hm'])
                data_dict['cls_preds_vina'] = torch.cat(cur_cls_preds_vina, dim=1)
                data_dict['cls_preds_vina'] = data_dict['cls_preds_vina'].permute(0, 2, 3, 1)

            # self.time_meter.update('centerhead post processing', (time.time() - postpro_time)*1000)

        return data_dict
