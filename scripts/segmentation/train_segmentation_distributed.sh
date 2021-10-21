#!/bin/bash
#SBATCH -J TrainSegModels-FastSCNN_d.%j
#SBATCH -N 1
#SBATCH -o ../../outs/segmentation/TrainSegModels-FastSCNN_d.%j.out
#SBATCH -e ../../errs/segmentation/TrainSegModels-FastSCNN_d.%j.err
#SBATCH -t 12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:V100:4

# model=$1
# device=$2

train_cfg="configs/segmentation/base_trainer.yaml"
model="FastSCNN"
img_dir="data/segmentation/Ego2Hands/train_imgs"  #  "../imgs/temp/train"
bg_dir="data/segmentation/Ego2Hands/bg_imgs" # "../imgs/temp/bg"   
log_dir="logs/segmentation/FastSCNN_d"
ckpt_dir="ckpts/segmentation/FastSCNN_d"
epochs=5 # 10
device="all"
resume_ckpt="ckpts/segmentation/FastSCNN_d/FastSCNN_epoch_10.pth"
resume_epochs=10
n_classes=3
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
    --n_classes $n_classes \
    --in_channels $in_channels \
    --distributed True \
    # --resume False \
    # --resume_ckpt $resume_ckpt \
    # --resume_epochs $resume_epochs \

