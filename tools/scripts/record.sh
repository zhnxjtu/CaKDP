# Calculate pruning criterion
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 gap_cal.py --launcher pytorch --batch_size 64 --cfg_file cfgs/kitti_pruning_measure/second.yaml --pretrained_model pretrained_model/kitti/second/checkpoint_epoch_80.pth --criterion_dir measure_doc --dataset_name kitti

# Training with IOU loss and KD loss
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 train_with_pruning.py --launcher pytorch --cfg_file cfgs/kitti_cakdp/second-voxel-pruning-x0.75.yaml --extra_tag second-voxel-0.75-v1

