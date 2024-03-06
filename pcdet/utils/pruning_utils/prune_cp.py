import torch
import numpy as np
from pcdet.models import build_network

def pruning(cfg, model_tmp, train_set):
    # TODO: calculate pruning index
    gap = np.loadtxt(cfg.MEA_FILE)
    gap = torch.tensor(gap).cuda()
    if gap.dim() > 1:
        gap = gap.sum(0)

    gap_v, gap_index = torch.sort(gap, descending=True)

    rate = cfg.PRUNING_RATE
    if rate < 1.0:
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
        filter_idx_2d = [i + 640 for i in
                         [0, 128, 256, 384, 512, 640, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816]]
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

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
        # print(model_new)
        # print( cfg.MODEL.BACKBONE_2D.NUM_FILTERS)
        # exit()

        if cfg.INHERIT:
            interval = 3
            count_3D = 0
            count_2D = 0
            for key, params in model_tmp.named_parameters():

                if key in model.state_dict():
                    if key.split('.')[0] == 'backbone_3d':
                        cur_idx = retain_index['3D_Conv'][count_3D // interval]
                        if key.split('.')[1] == 'conv_input':
                            model.state_dict()[key].copy_(params.data[cur_idx])
                        else:
                            front_idx = retain_index['3D_Conv'][(count_3D // interval) - 1]
                            if params.data.dim() == 5:
                                model.state_dict()[key].copy_(params.data[cur_idx, ...][..., front_idx])
                            elif params.data.dim() == 1:
                                model.state_dict()[key].copy_(params.data[cur_idx])
                            else:
                                assert False

                        count_3D += 1

                    if key.split('.')[0] == 'backbone_2d':
                        if count_2D == 0:
                            front_idx = retain_index['3D_Conv'][11].tolist() + [i + 128 for i in
                                                                                retain_index['3D_Conv'][11]]
                        cur_idx = retain_index['2D_Conv'][count_2D // interval]
                        if key.split('.')[1] == 'blocks':
                            if params.data.dim() == 4:
                                model.state_dict()[key].copy_(params.data[cur_idx, ...][:, front_idx, ...])
                                front_idx = cur_idx
                            elif params.data.dim() == 1:
                                model.state_dict()[key].copy_(params.data[cur_idx])
                            else:
                                assert False

                        if key.split('.')[1] == 'deblocks':
                            if params.data.dim() == 4:
                                if key.split('.')[2] == '0':
                                    model.state_dict()[key].copy_(
                                        params.data[:, cur_idx, ...][retain_index['2D_Conv'][5], ...])
                                elif key.split('.')[2] == '1':
                                    model.state_dict()[key].copy_(
                                        params.data[:, cur_idx, ...][retain_index['2D_Conv'][11], ...])
                                else:
                                    assert False
                            elif params.data.dim() == 1:
                                model.state_dict()[key].copy_(params.data[cur_idx])
                            else:
                                assert False

                        count_2D += 1

                    if (key.split('.')[0] == 'dense_head') and (key.split('.')[1] == 'shared_conv'):
                        front_idx = retain_index['2D_Conv'][12].tolist() + [i + 256 for i in
                                                                            retain_index['2D_Conv'][13]]
                        if params.data.dim() == 4:
                            model.state_dict()[key].copy_(params.data[:, front_idx, ...])
                        elif params.data.dim() == 1:
                            model.state_dict()[key].copy_(params.data)
                        else:
                            assert False

    else:
        model = model_tmp

    return model, cfg



