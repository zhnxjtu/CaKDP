# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 


## Dataset Preparation
Please follow the OpenPCDet [tutorial](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) to 
prepare needed datasets.

## Training & Testing
[//]: # ( TODO)
### Step 1: Train the models (SECOND and PV-RCNN as examples)
```shell
(4 GPUs, SECOND)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 18891 --nproc_per_node=4 train.py --cfg_file cfgs/kitti_models/second.yaml --launcher pytorch --extra_tag baseline
(4 GPUs, PV-RCNN)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 18891 --nproc_per_node=4 train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --launcher pytorch --extra_tag baseline
```
Here, we provide pretrained models [link](https://drive.google.com/drive/folders/1VTSrXW8MiW_1kbxZEPEMTxq8ZIXUVzuw?usp=sharing). Please put the pretrained model in '/tools/pretrained_model/[#dataset]/[#model name]/[#]'.





### Step 2: Distillation (CP-Pillar-v0.4 as example)
Modify following keys in the student distillation config
```yaml
# cfgs/waymo_models/cp-pillar/cp-pillar-v0.4_sparsekd.yaml
TEACHER_CKPT: ${PATH_TO_TEACHER_CKPT}
PRETRAINED_MODEL: ${PATH_TO_TEACHER_CKPT}
```
Run the training config
```shell
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/waymo_models/cp-pillar/cp-pillar-v0.4_sparsekd.yaml 
```

## Calculate Efficiency Metrics

### Prepare
Make sure you have installed our customized Thop as [INSTALL.md](./INSTALL.md).
To calculate the Flops and Acts for spconv-based models, you also need to replace original `conv.py` in spconv
with our modified one.
```shell
# replace our modified conv file for 
# make sure your spconv is at least 2.1.20
cp extra_files/conv.py ${CONDA_PATH}/envs/${ENV_NAME}/lib/${PYTHON_VERSION}/site-packages/spconv/pytorch/
```

### Command
```shell
# Take Waymo as an example
# This command have to be executed on single gpu only
python test.py --cfg_file ${CONFIG_PATH} --batch_size 1 --ckpt ${CKPT_PATH} --infer_time --cal_params \
  --set DATA_CONFIG.DATA_SPLIT.test infer_time DATA_CONFIG.SAMPLED_INTERAVL.test 2
``` 

