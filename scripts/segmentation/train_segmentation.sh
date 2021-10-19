#!/bin/bash
#SBATCH -J TrainSegModels.%j
#SBATCH -N 1
#SBATCH -o ../outs/segmentation/TrainSegModels-FastSCNN.%j.out
#SBATCH -e ../errs/segmentation/TrainSegModels-FastSCNN.%j.err
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:V100:1

# model=$1
# device=$2

train_cfg="configs/segmentation/base_trainer"
model="FastSCNN"
img_dir="data/segmentation/Ego2Hands/train_imgs"
bg_dir="data/segmentation/Ego2Hands/bg_imgs"
log_dir="logs/segmentation/fastscnn"
ckpt_dir="ckpts/segmentation/fastscnn"
epochs=10
device="0"
lr=None
distributed=False
resume=False
resume_ckpt="ckpts/segmentation/fastscnn/FastSCNN_epoch_10.pth"
resume_epochs=10
n_classes=2
in_channels=1

cd ../..
module load nvidia/10.2

# python -m realtime_hand_3d.segmentation.seg_trainer \
python realtime_hand_3d/segmentation/seg_trainer.py \
    --train_cfg $train_cfg \
    --model $model \
    --img_dir $img_dir \
    --bg_dir $bg_dir \
    --log_dir $log_dir \
    --ckpt_dir $ckpt_dir \
    --epochs $epochs \
    --device $device \
    --lr $lr \
    --distributed $distributed \
    --resume $resume \
    --resume_ckpt $resume_ckpt \
    --resume_epochs $resume_epochs \
    --n_classes $n_classes \
    --in_channels $in_channels 
