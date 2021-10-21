#!/bin/bash
#SBATCH -J TrainSegModels-ICNet.%j
#SBATCH -N 1
#SBATCH -o ../../outs/segmentation/TrainSegModels-ICNet.%j.out
#SBATCH -e ../../errs/segmentation/TrainSegModels-ICNet.%j.err
#SBATCH -t 12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:V100:1

# model=$1
# device=$2

train_cfg="configs/segmentation/base_trainer.yaml"
model="ICNet"
img_dir="data/segmentation/Ego2Hands/train_imgs"  #  "../imgs/temp/train"
bg_dir="data/segmentation/Ego2Hands/bg_imgs" # "../imgs/temp/bg"   
log_dir="logs/segmentation/ICNet"
ckpt_dir="ckpts/segmentation/ICNet"
epochs=5 # 10
device="0"
resume_ckpt="ckpts/segmentation/ICNet/ICNet_epoch_10.pth"
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
    --in_channels $in_channels 
    # --distributed False \
    # --resume False \
    # --resume_ckpt $resume_ckpt \
    # --resume_epochs $resume_epochs \
