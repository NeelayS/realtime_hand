#!/bin/bash
#SBATCH -J TestSegModelsInfSpeed.%j
#SBATCH -N 1
#SBATCH -o ../../outs/segmentation/TestSegModelsInfSpeed.out
#SBATCH -e ../../errs/segmentation/TestSegModelsInfSpeed.err
#SBATCH -t 00:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:V100:1

module load nvidia/10.2
cd ../..

python realtime_hand_3d/segmentation/test_inference.py --video "data/segmentation/test.mp4" --device "cpu" --all_models True --inp_size 512 
                          