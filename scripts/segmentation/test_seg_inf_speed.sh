#!/bin/bash
#SBATCH -J TestSegModelsInfSpeed_V100.%j
#SBATCH -N 1
#SBATCH -o ../../outs/segmentation/TestSegModelsInfSpeed_V100.out
#SBATCH -e ../../errs/segmentation/TestSegModelsInfSpeed_V100.err
#SBATCH -t 00:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:V100:1

module load nvidia/10.2
cd ../..

python -m realtime_hand_3d.segmentation.test_inference --video "../data/test.mp4" --device "cuda:0" --inp_size 512 --model "BiSeNet" #--all_models 
                          