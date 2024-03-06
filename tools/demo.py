import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list



    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):

        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ckpt_teacher', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg



def main():

    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # demo_dataset = DemoDataset(
    #     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), ext=args.ext, logger=logger
    # )
    #
    # logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=1, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(test_loader):
            # if '000124' in data_dict['frame_id']:
            if (idx+1) == 66:
            # if idx > 260:

                logger.info(f'Visualized sample index: \t{idx + 1}')
                # data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)
                gt_boxes = data_dict['gt_boxes'].squeeze(0)[:,:-1]
                box_preds = pred_dicts[0]['pred_boxes']

                cur_gt = gt_boxes
                k = cur_gt.__len__() - 1
                while k > 0 and cur_gt[k].sum() == 0:
                    k -= 1
                cur_gt = cur_gt[:k + 1]
                print(data_dict['gt_boxes'].squeeze(0)[:,-1])
                print(pred_dicts[0]['pred_labels'])


                # if cur_gt.shape[0] > 0:
                #     if box_preds.shape[0] > 0:
                #         iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
                #     else:
                #         iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))
                #
                # iou, iou_idx = iou3d_rcnn.max(dim=0)
                # box_preds_selected = box_preds[iou_idx]
                # box_scores_selected = pred_dicts[0]['pred_scores'][iou_idx]
                # box_labels_selected = pred_dicts[0]['pred_labels'][iou_idx]
                # print(box_preds_selected)
                # print(box_scores_selected)
                # print(cur_gt[:, 0:7])
                # print('OBJ_NUM:', cur_gt.shape[0])


                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:],
                #     gt_boxes=gt_boxes,
                #     ref_boxes=box_preds_selected,
                #     ref_scores=box_scores_selected,
                #     ref_labels=box_labels_selected,
                #     draw_origin=True
                # )
                # print(pred_dicts[0]['pred_scores'].shape[0])
                # print(gt_boxes)
                # print(gt_boxes[1,3]*gt_boxes[1,4])
                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    gt_boxes=gt_boxes,
                    ref_boxes=box_preds,
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    draw_origin=True
                )

                # min_idx = box_scores_selected.unsqueeze(0).min(dim=1)[1]
                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:],
                #     gt_boxes=gt_boxes,
                #     ref_boxes=box_preds_selected[min_idx],
                #     ref_scores=box_scores_selected[min_idx],
                #     ref_labels=box_labels_selected[min_idx],
                #     draw_origin=True
                # )

                if not OPEN3D_FLAG:
                    mlab.show(stop=True)

                # break

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
