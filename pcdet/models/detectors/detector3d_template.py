import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads, kd_adapt_block, kd_trans_block
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe, pillar_adaptor
from ..model_utils import model_nms_utils
from ...utils.kd_utils import kd_tgi_utils

try:
    from torch.nn.modules.conv import _ConvTransposeNd as _ConvTransposeNd
except:
    from torch.nn.modules.conv import _ConvTransposeMixin as _ConvTransposeNd


class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pillar_adaptor', 'pfe',
            'backbone_2d', 'dense_head', 'dense_head_aux', 'kd_adapt_block', 'point_head', 'roi_head', 'kd_trans_block'
        ]
        # For kd only
        self.is_teacher = False

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size_tea if hasattr(self.dataset, 'grid_size_tea') and self.model_cfg.get('IS_TEACHER', None) else self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size_tea if hasattr(self.dataset, 'voxel_size_tea') and self.model_cfg.get('IS_TEACHER', None) else self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_pillar_adaptor(self, model_info_dict):
        if self.model_cfg.get('PILLAR_ADAPTOR', None) is None:
            return None, model_info_dict

        pillar_adapt_module = pillar_adaptor.__all__[self.model_cfg.PILLAR_ADAPTOR.NAME](
            model_cfg=self.model_cfg.PILLAR_ADAPTOR,
            in_channel=model_info_dict['num_bev_features'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(pillar_adapt_module)
        return pillar_adapt_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_kd_adapt_block(self, model_info_dict):
        if self.model_cfg.get('KD_ADAPT_BLOCK', None) is None:
            return None, model_info_dict

        kd_adapt_block_module = kd_adapt_block.__all__[self.model_cfg.KD_ADAPT_BLOCK.NAME](
            model_cfg=self.model_cfg.KD_ADAPT_BLOCK,
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(kd_adapt_block_module)
        return kd_adapt_block_module, model_info_dict



    def build_kd_trans_block(self, model_info_dict):

        if self.model_cfg.get('KD_TRANS_BLOCK', None) is None:
            return None, model_info_dict

        kd_trans_block_module = kd_trans_block.__all__[self.model_cfg.KD_TRANS_BLOCK.NAME](
            model_cfg=self.model_cfg.KD_TRANS_BLOCK,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        # print('aaaaaaaaaaaaaaaaa')
        # print(model_info_dict['module_list'])
        model_info_dict['module_list'].append(kd_trans_block_module)
        return kd_trans_block_module, model_info_dict

    # def build_kd_trans_block(self, model_info_dict):
    #
    #     if self.model_cfg.get('KD_TRANS_BLOCK', None) is None:
    #         return None, model_info_dict
    #
    #     kd_trans_block_module = kd_trans_block.__all__[self.model_cfg.KD_TRANS_BLOCK.NAME](
    #         model_cfg=self.model_cfg.KD_TRANS_BLOCK,
    #         point_cloud_range=model_info_dict['point_cloud_range'],
    #         voxel_size=model_info_dict['voxel_size'],
    #         backbone_channels=model_info_dict['backbone_channels']
    #     )
    #     # print('aaaaaaaaaaaaaaaaa')
    #     # print(model_info_dict['module_list'])
    #     model_info_dict['module_list'].append(kd_trans_block_module)
    #
    #     return kd_trans_block_module, model_info_dict



    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False) or self.model_cfg.get('KD_TRANS_BLOCK', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict
    
    def build_dense_head_aux(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_AUX', None) is None:
            return None, model_info_dict
        dense_head_aux_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD_AUX.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD_AUX,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD_AUX.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_aux_module)
        return dense_head_aux_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            backbone_channels=model_info_dict['backbone_channels'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []


        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if batch_dict.get('batch_iou_preds', None) is not None:

                fea_3d = batch_dict['spatial_features'][batch_mask]
                fea_3d = fea_3d.abs().sum(0)
                fea_mask = fea_3d > 0

                iou_preds = batch_dict['batch_iou_preds'][batch_mask]

                # iou_preds = iou_preds * fea_mask.unsqueeze(-1)
                iou_preds = iou_preds.view(-1, 3)
                iou_preds = torch.sigmoid(iou_preds)
                

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    if batch_dict.get('batch_iou_preds', None) is not None:
                        cls_preds = torch.sigmoid(cls_preds) * (iou_preds.max(-1)[0].unsqueeze(-1) > post_process_cfg.POST_IOU_THRESH).float()
                        # cls_preds = torch.sigmoid(cls_preds) * (iou_preds.max(-1)[0].unsqueeze(-1) >= 0.0).float()
                        # cls_preds = torch.sigmoid(cls_preds) * (iou_preds.max(-1)[0].unsqueeze(-1)).float()
                    else:
                        cls_preds = torch.sigmoid(cls_preds)

            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1

                '''
                print(label_preds.max())
                exit()
                '''

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                # if batch_dict.get('batch_iou_preds', None) is not None:
                #     selected, selected_scores = model_nms_utils.class_agnostic_nms_change(
                #         box_scores=cls_preds, box_preds=box_preds, iou_preds=iou_preds.squeeze(),
                #         nms_config=post_process_cfg.NMS_CONFIG,
                #         score_thresh=post_process_cfg.SCORE_THRESH
                #     )
                # else:
                #     selected, selected_scores = model_nms_utils.class_agnostic_nms(
                #         box_scores=cls_preds, box_preds=box_preds,
                #         nms_config=post_process_cfg.NMS_CONFIG,
                #         score_thresh=post_process_cfg.SCORE_THRESH
                #     )

                # no sigmoid of cls response
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

                # print(final_scores.shape)
                # print(final_labels.shape)
                # print(final_boxes.shape)
                # exit()


            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )


            # TODO: test distance
            '''
            import numpy as np
            cur_gt = batch_dict['gt_boxes'][index]
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]

            if cur_gt.shape[0] > 0:
                if final_boxes.shape[0] > 0:
                    iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(final_boxes[:, 0:7], cur_gt[:, 0:7])
                    iou, iou_idx = iou3d_rcnn.max(dim=0)
                    find_zero = (iou==0)
                    if find_zero.sum() != 0:
                        zero_idx = torch.nonzero(find_zero).squeeze()
                        print('a', cur_gt[:, 7][zero_idx])
                        print('b', final_labels[iou_idx][zero_idx])

                        print('c', cur_gt[zero_idx])
                        print('d', final_labels[iou_idx])

                        gap = batch_dict['points'][:, 1:4] - cur_gt[zero_idx][0:3]
                        dist = (gap[:,0].pow(2) + gap[:,1].pow(2)).pow(0.5)
                        a = (dist<3.5)
                        print(a.sum())
                        print(dist.topk(5, largest=False))
                        print(batch_dict['points'][dist.topk(5, largest=False)[1]])

                        print(dist.unsqueeze(-1).min(dim=0))
                        print(batch_dict['points'][dist.unsqueeze(-1).min(dim=0)[1]])

                        exit()
            '''


                        # with open('undetect_gt.txt', 'a', encoding='utf-8') as file:
                        #     np.set_printoptions(suppress=True)
                        #     if cur_gt[zero_idx].dim() == 1:
                        #         sa = str(cur_gt[zero_idx].detach().cpu().numpy())
                        #         sa = sa.replace('[',' ').replace(']',' ').replace('\n',' ')
                        #         file.write(sa + '\n')
                        #     else:
                        #         for cnt in range(cur_gt[zero_idx].shape[0]):
                        #             cur_gt_tmp = cur_gt[zero_idx].detach().cpu()
                        #             sa = str(cur_gt_tmp[cnt].numpy())
                        #             sa = sa.replace('[',' ').replace(']',' ').replace('\n',' ')
                        #             file.write(sa + '\n')

                        # print('**'*10)
                        # print(batch_dict['points'].max(0)[0])
                        # print(batch_dict['points'].min(0)[0])

                        # exit()


            # print('a', cur_gt[:, 7])
            # print('b', final_labels[iou_idx])
            #
            #
            # exit()



            # # TODO: plot the distribution of the results
            # if not self.is_teacher:
            #     cur_gt = batch_dict['gt_boxes'][index]
            #     k = cur_gt.__len__() - 1
            #     while k > 0 and cur_gt[k].sum() == 0:
            #         k -= 1
            #     cur_gt = cur_gt[:k + 1]
            #
            #     if cur_gt.shape[0] > 0:
            #         if final_boxes.shape[0] > 0:
            #             iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(final_boxes[:, 0:7], cur_gt[:, 0:7])
            #
            #             iou, iou_idx = iou3d_rcnn.max(dim=0)
            #             for c in range(int(iou.shape[0])):
            #                 with open('PV-rcnn-iou.txt', 'a') as file:
            #                     file.write(str(iou[c].detach().cpu().item()) + '\n')
            #                 # print(iou_idx[c])
            #                 # print(final_scores)
            #
            #                 cur_sco = final_scores[iou_idx[c]]
            #                 with open('PV-rcnn-score.txt', 'a') as file2:
            #                     file2.write(str(cur_sco.detach().cpu().item()) + '\n')



            # exit()

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])


                # print('iou:', iou3d_rcnn.max(dim=0)[0])
                # print('iou_idx', iou3d_rcnn.max(dim=0)[1])
                # with open('raw-x0.5.txt', 'a') as file:
                #     file.write('iou:' + str(iou3d_rcnn.max(dim=0)[0]) + '\n')
                #     file.write('iou_idx:' + str(iou3d_rcnn.max(dim=0)[1]) + '\n')
                # exit()


            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                if self.model_cfg.get('IGNORE_PRETRAIN_MODULES', None):
                    module_name = key.split('.')[0]
                    if module_name in self.model_cfg.IGNORE_PRETRAIN_MODULES:
                        continue
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, remap_cfg=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)
        if remap_cfg and remap_cfg.ENABLED:
            logger.info('==> Remap pretrained model parameters with: %s' % remap_cfg.WAY.lower())
            model_state_disk = self._remap_to_current_model(model_state_disk, remap_cfg)
        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    def get_kd_loss(self, batch_dict, tb_dict, disp_dict):
        kd_loss, tb_dict = self.kd_head.get_kd_loss(batch_dict, tb_dict)
        # pillar adaptor kd loss
        if self.pillar_adaptor is not None and self.pillar_adaptor.cal_loss:
            kd_pillar_loss, tb_dict = self.pillar_adaptor.get_loss(batch_dict, tb_dict)
            kd_loss += kd_pillar_loss
        # vfe kd loss
        if self.model_cfg.get('VFE_KD', None):
            vfe_kd_loss, tb_dict = self.kd_head.get_vfe_kd_loss(
                batch_dict, tb_dict, self.model_cfg.KD_LOSS.VFE_LOSS
            )
            kd_loss += vfe_kd_loss

        # roi kd loss
        if self.model_cfg.get('ROI_KD', None):
            roi_kd_loss, tb_dict = self.kd_head.get_roi_kd_loss(batch_dict, tb_dict)
            kd_loss += roi_kd_loss

        disp_dict.update({
            'kd_ls': '{:.2f}'.format(kd_loss if isinstance(kd_loss, float) else kd_loss.item())
        })
        for key, val in tb_dict.items():
            if 'kd' in key:
                disp_dict[key] = '{:.2f}'.format(val if isinstance(val, float) else val.item())

        return kd_loss, tb_dict, disp_dict

    def _remap_to_current_model(self, model_state, cfg=None):
        return getattr(kd_tgi_utils, '_remap_to_current_model_by_{}'.format(cfg.WAY.lower()))(self, model_state, cfg)