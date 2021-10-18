#!/bin/bash
#SBATCH -J TestSegModelsInfSpeed.%j
#SBATCH -N 1
#SBATCH -o ../outs/segmentation/TestSegModelsInfSpeed.%j.out
#SBATCH -e ../errs/segmentation/TestSegModelsInfSpeed.%j.err
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:V100:1

module load nvidia/10.2
cd ../realtime_hand_3d/segmentation

python test_inference.py --video ../../data/segmentation/test.mp4 \
                         # --model UNet \
                         --all_models True \  
                         --device 0 \
                         --inp_size 512 \
                         --viz False \
                         --out_dir "." \
                          