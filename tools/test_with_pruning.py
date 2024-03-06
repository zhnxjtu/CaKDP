import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, build_teacher_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--infer_time', action='store_true', default=False, help='')
    parser.add_argument('--cal_params', action='store_true', default=False, help='')
    parser.add_argument('--teacher_ckpt', type=str, default=None, help='checkpoint to start from')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


def train_eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, ckpt_dir, dist_test=False):
    # load checkpoint
    filename = os.path.join(ckpt_dir, 'checkpoint_epoch_%s.pth' % epoch_id)
    model.load_params_from_file(filename=filename, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=False
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )



    # TODO: calculate pruning index
    if cfg.get('MEA_FILE'):
        gap = np.loadtxt(cfg.MEA_FILE)
        gap = torch.tensor(gap).cuda()
        if gap.dim() > 1:
            gap = gap.sum(0)

        gap_v, gap_index = torch.sort(gap, descending=True)

        rate = cfg.PRUNING_RATE
        retain_index_tmp = gap_index[:np.int(rate * len(gap))]

        # 3D Conv
        # num_filter: [16, 16, 32, 64, 64, 128]
        # num_layer: [1, 1, 3, 3, 3, 1]
        # 2D Conv
        # num_filter: [128, 256, 256, 256]
        # num_layer: [6, 6, 1, 1]

        retain_index = {
            '3D_Conv': {},
            '3D_Conv_num': torch.Tensor(12),
            '2D_Conv': {},
            '2D_Conv_num': torch.Tensor(14),
        }

        # num_filter = [16, 16, 32, 32, 32, 64, 64, 64, 64, 64, 64, 128]
        filter_idx_3d = [0, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 640]
        retain_3d_num = 0
        for layer_c in range(len(filter_idx_3d) - 1):
            idx_cur_layer = ((retain_index_tmp < filter_idx_3d[layer_c + 1]) &
                             (retain_index_tmp >= filter_idx_3d[layer_c])).nonzero().squeeze(-1)
            if idx_cur_layer.dim() > 0:
                retain_index['3D_Conv'][layer_c] = retain_index_tmp[idx_cur_layer] - filter_idx_3d[layer_c]
                retain_3d_num += len(retain_index['3D_Conv'][layer_c])
                retain_index['3D_Conv_num'][layer_c] = len(retain_index['3D_Conv'][layer_c])
            else:
                assert False

        # num_filter = [128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256]
        filter_idx_2d = [i + 640 for i in [0, 128, 256, 384, 512, 640, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816]]
        retain_2d_num = 0
        for layer_c in range(len(filter_idx_2d) - 1):
            idx_cur_layer = ((retain_index_tmp < filter_idx_2d[layer_c + 1]) &
                             (retain_index_tmp >= filter_idx_2d[layer_c])).nonzero().squeeze(-1)
            if idx_cur_layer.dim() > 0:
                retain_index['2D_Conv'][layer_c] = retain_index_tmp[idx_cur_layer] - filter_idx_2d[layer_c]
                retain_2d_num += len(retain_index['2D_Conv'][layer_c])
                retain_index['2D_Conv_num'][layer_c] = len(retain_index['2D_Conv'][layer_c])
            else:
                assert False

        print('*' * 20)
        print(
            '\n pruned_filter_num: {} / {} ({}), pruned_3D_conv_num: {} / {} ({}), pruned_2D_conv_num: {} / {} ({}). \n'.format(
                (filter_idx_2d[-1] - (retain_3d_num + retain_2d_num)), filter_idx_2d[-1],
                ((filter_idx_2d[-1] - (retain_3d_num + retain_2d_num)) / filter_idx_2d[-1]),
                (filter_idx_3d[-1] - retain_3d_num), filter_idx_3d[-1],
                ((filter_idx_3d[-1] - retain_3d_num) / filter_idx_3d[-1]),
                (filter_idx_2d[-1] - filter_idx_3d[-1] - retain_2d_num), (filter_idx_2d[-1] - filter_idx_3d[-1]),
                ((filter_idx_2d[-1] - filter_idx_3d[-1] - retain_2d_num) / (filter_idx_2d[-1] - filter_idx_3d[-1])),
            ))
        print('*' * 20)
        # print(retain_index['3D_Conv_num'].int().tolist())

        cfg.MODEL.BACKBONE_3D.NUM_FILTERS = retain_index['3D_Conv_num'].int().tolist()
        cfg.MODEL.BACKBONE_3D.NAME = 'VoxelBackBone8x_flexible'
        cfg.MODEL.BACKBONE_2D.IN_CHANNEL = 2 * cfg.MODEL.BACKBONE_3D.NUM_FILTERS[-1]
        cfg.MODEL.BACKBONE_2D.NAME = 'BaseBEVBackbone_flexible'
        cfg.MODEL.BACKBONE_2D.NUM_FILTERS = retain_index['2D_Conv_num'].int().tolist()

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    logger.info(model)

    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
