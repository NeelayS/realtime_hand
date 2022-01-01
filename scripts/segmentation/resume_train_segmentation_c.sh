#!/bin/bash
#SBATCH -J ResumeTrainSegModels-SwiftNetRes18.%j
#SBATCH -N 1
#SBATCH -o ../../../results/outs/segmentation/training/SwiftNetRes18.%j.out
#SBATCH -e ../../../results/errs/segmentation/training/SwiftNetRes18.%j.err
#SBATCH -t 36:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:V100:1

# model=$1
# device=$2

n_classes=3
in_channels=2 # 3

train_cfg="configs/segmentation/custom_loss_trainer.yaml"
model="SwiftNetRes18"
img_dir="../data/sub_imgs" # "data/segmentation/Ego2Hands/train_imgs/" 
bg_dir="data/segmentation/Ego2Hands/bg_imgs" 
log_dir="../results/logs/segmentation/SwiftNetRes18/run1"
ckpt_dir="../results/ckpts/segmentation/SwiftNetRes18/run1"
epochs=20
device="0"

resume_ckpt="../results/ckpts/segmentation/SwiftNetRes18/run1/SwiftNetResNet_epochs23.pth"
resume_epochs=15

cd ../..
module load nvidia/10.2

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
    --in_channels $in_channels \
    --resume \
    --resume_ckpt $resume_ckpt \
    --resume_epochs $resume_epochs \

