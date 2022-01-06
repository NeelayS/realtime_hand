#!/bin/bash
#SBATCH -J TrainSegModels-ModBiSeNet.%j
#SBATCH -N 1
#SBATCH -o ../../../results/outs/segmentation/training/ModBiSeNet.%j.out
#SBATCH -e ../../../results/errs/segmentation/training/ModBiSeNet.%j.err
#SBATCH -t 36:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:V100:1

# model=$1
# device=$2

train_cfg="configs/segmentation/custom_loss_trainer.yaml"
model="ModBiSeNet"
img_dir="../data/sub_imgs" # "data/segmentation/Ego2Hands/train_imgs/" 
bg_dir="data/segmentation/Ego2Hands/bg_imgs" 
log_dir="../results/logs/segmentation/ModBiSeNet/run1"
ckpt_dir="../results/ckpts/segmentation/ModBiSeNet/run1"
epochs=20
device="0"

resume_ckpt="../results/ckpts/segmentation/ModBiSeNet/ModBiSeNet_epoch_10.pth"
resume_epochs=10
n_classes=3
in_channels=2 # 3

module load nvidia/10.2
cd ../..

python -m realtime_hand_3d.segmentation.seg_trainer \
    --train_cfg $train_cfg \
    --model $model \
    --img_dir $img_dir \
    --bg_dir $bg_dir \
    --log_dir $log_dir \
    --ckpt_dir $ckpt_dir \
    --epochs $epochs \
    --device $device \
    --n_classes $n_classes \
    --in_channels $in_channels 
    # --distributed False \
    # --resume False \
    # --resume_ckpt $resume_ckpt \
    # --resume_epochs $resume_epochs \

