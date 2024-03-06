import pickle
import time

import torch
import tqdm
import copy
import numpy as np

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.kd_utils import kd_forwad
import os
from pathlib import Path

try:
    from thop import clever_format
except:
    pass
    # you cannot use cal_param without profile


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False,
                   result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    # To be compatible with train eval
    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()
    
    if getattr(args, 'cal_params', False):
        flops_meter = common_utils.AverageMeter()
        # acts_meter = common_utils.AverageMeter()
        params_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    # model.apply(common_utils.set_bn_train)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            # pred_dicts, ret_dict = model(batch_dict, record_time=getattr(args, 'infer_time', False) and i > start_iter)
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        if getattr(args, 'infer_time', False) and i > start_iter:
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        if getattr(args, 'cal_params', False):
            # macs, params, acts = common_utils.cal_flops(model, batch_dict)
            macs, params = common_utils.cal_flops(model, batch_dict)
            flops_meter.update(macs)
            params_meter.update(params)

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if getattr(args, 'cal_params', False):
        # macs, params, acts = clever_format([flops_meter.avg, params, acts_meter.avg], "%.3f")
        # print(f'\nparams: {params}\nmacs: {macs}\nacts: {acts}\n')
        macs, params = clever_format([flops_meter.avg, params_meter.avg], "%.3f")
        print(f'\nparams: {params}\nmacs: {macs}\n')

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if getattr(args, 'infer_time', False):
        print(model.module_time_meter)
        if hasattr(model.dense_head, 'time_meter'):
            print(model.dense_head.time_meter)

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    if cfg.MODEL.POST_PROCESSING.get('EVAL_CLASSES', None):
        result_dict, result_str = get_multi_classes_mAP(
            result_dict, result_str, cfg.MODEL.POST_PROCESSING.EVAL_CLASSES
        )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_one_epoch_measurement_plain(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False,
                   result_dir=None, criterion_file_dir=None, dataset_name=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    # model.apply(common_utils.set_bn_train)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    err_all = 0.0
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        print('The %d-th sample' %i)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict, generate_dict = model(batch_dict, record_time=getattr(args, 'infer_time', False) and i > start_iter)
            if model.module.__class__.__name__ == 'CenterPoint':
                cls_num = generate_dict['cls_preds_vina'].shape[-1]
                batch_size = generate_dict['cls_preds_vina'].shape[0]
                pred_tmp = generate_dict['cls_preds_vina'].reshape(batch_size, -1, cls_num)
            else:
                cls_num = generate_dict['batch_cls_preds'].shape[-1]
                batch_size = generate_dict['batch_size']
                pred_tmp = generate_dict['cls_preds_vina'].view(batch_size, -1, cls_num)


            # pruning
            err_list = []
            model_tmp = copy.deepcopy(model)
            if dist_test:
                model_tmp = model_tmp.module
            for key, params in model_tmp.named_parameters():
                if key.split('.')[0] in ['backbone_2d', 'backbone_3d']:
                    if params.dim() == 5:
                        c_out, h1, h2, h3, c_in = params.shape
                        for f_count in range(c_out):
                            para_save = params[f_count, ...].clone()
                            params[f_count, ...] = 0.0
                            pred_dicts_, ret_dict_, generate_dict_ = model_tmp(batch_dict)
                            pred_cur = generate_dict_['cls_preds_vina'].view(batch_size, -1, cls_num)
                            err = ((torch.sigmoid(pred_cur) - torch.sigmoid(pred_tmp)).abs().sum() / (pred_cur.shape[1]))
                            params[f_count, ...] = para_save.clone()
                            err_list.append(err)
                    elif params.dim() == 4:
                        if key.split('.')[1] == 'blocks':
                            c_out, c_in, h1, h2, = params.shape
                            for f_count in range(c_out):
                                para_save = params[f_count, ...].clone()
                                params[f_count, ...] = 0.0
                                pred_dicts_, ret_dict_, generate_dict_ = model_tmp(batch_dict)
                                pred_cur = generate_dict_['cls_preds_vina'].view(batch_size, -1, cls_num)
                                err = ((torch.sigmoid(pred_cur) - torch.sigmoid(pred_tmp)).abs().sum() / (pred_cur.shape[1]))
                                params[f_count, ...] = para_save.clone()
                                err_list.append(err)
                        elif key.split('.')[1] == 'deblocks':
                            c_in, c_out, h1, h2, = params.shape
                            for f_count in range(c_out):
                                para_save = params[:, f_count, ...].clone()
                                params[:, f_count, ...] = 0.0
                                pred_dicts_, ret_dict_, generate_dict_ = model_tmp(batch_dict)
                                pred_cur = generate_dict_['cls_preds_vina'].view(batch_size, -1, cls_num)
                                err = ((torch.sigmoid(pred_cur) - torch.sigmoid(pred_tmp)).abs().sum() / (pred_cur.shape[1]))
                                params[:, f_count, ...] = para_save.clone()
                                err_list.append(err)

            err_all += torch.tensor(err_list)

        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    # TODO: get pruning criterion
    model_name = model.module.__class__.__name__
    criterion_file_dir_ = os.getcwd() + '/' + criterion_file_dir
    Path(criterion_file_dir_).mkdir(parents=True, exist_ok=True)
    with open(criterion_file_dir_ + '/' + model_name + '_' + dataset_name + '.txt', 'a') as file:
        np.set_printoptions(suppress=True, threshold=1e8)
        save_measure = str(np.array(err_all.detach().cpu().numpy()))
        save_measure = save_measure.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
        file.write(save_measure + '\n')

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    if cfg.MODEL.POST_PROCESSING.get('EVAL_CLASSES', None):
        result_dict, result_str = get_multi_classes_mAP(
            result_dict, result_str, cfg.MODEL.POST_PROCESSING.EVAL_CLASSES
        )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    print('********* Measurement has been completed *********')
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def get_multi_classes_mAP(result_dict, result_str, metric_dict):
    result_str += '\nmAP\n'
    for metric, class_list in metric_dict.items():
        mAP = 0
        for cls in class_list:
            mAP += result_dict[cls]
        mAP /= len(class_list)
        result_dict['mAP/' + metric] = mAP
        result_str += metric + ' mAP: {:.4f}\n'.format(mAP)

    return result_dict, result_str


if __name__ == '__main__':
    pass
