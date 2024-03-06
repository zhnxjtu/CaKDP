# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 


## Dataset Preparation
Please follow the OpenPCDet [tutorial](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to 
prepare needed datasets.

## Training & Testing
[//]: # ( TODO)
### Step 1: Train the models (SECOND and PV-RCNN as examples, on 4 GPUs)
```
(SECOND)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 18891 --nproc_per_node=4 train.py --cfg_file cfgs/kitti_models/second.yaml --launcher pytorch --extra_tag baseline
(PV-RCNN)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 18891 --nproc_per_node=4 train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --launcher pytorch --extra_tag baseline
```
(Here, we provide pretrained models ([link](https://drive.google.com/drive/folders/1VTSrXW8MiW_1kbxZEPEMTxq8ZIXUVzuw?usp=sharing)). Please put the pretrained model in "/tools/pretrained_model/{#dataset}/{#model name}/{#}".)

### Step 2: Prune the models (SECOND as example, on 8 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 gap_cal.py --launcher pytorch --batch_size 64 --cfg_file cfgs/kitti_pruning_measure/second.yaml --pretrained_model pretrained_model/kitti/second/checkpoint_epoch_80.pth --criterion_dir measure_doc --dataset_name kitti
```
(The measurement results are saved in "/tools/measure_doc/". Here, we provide the measurement result for SECOND.)

### Step 3: Distillation (PV-RCNN & SECOND-x0.75 as example, on 4 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 18891 --nproc_per_node=4 train.py --cfg_file cfgs/kitti_cakdp/second-pv-x0.75.yaml --launcher pytorch --extra_tag pv-second-x0.75
```
Please modify the "cif_file" and "extra_tag" to get results for other models.

###  Test without pruning
Please modify following "cif_file" and "model" first.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 18891 --nproc_per_node=4 test.py --cfg_file {#cfg_file} --ckpt {#model} --launcher pytorch
```
###  Test after pruning 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 18891 --nproc_per_node=4 test_with_pruning.py --cfg_file {#cfg_file} --ckpt {#model} --launcher pytorch
```

## Calculate Efficiency Metrics
Please refer to [SparseKD](https://github.com/CVMI-Lab/SparseKD/blob/master/docs/GETTING_STARTED.md).
