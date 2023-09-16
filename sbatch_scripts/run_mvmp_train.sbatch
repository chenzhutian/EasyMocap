#!/bin/bash
#SBATCH -p seas_gpu
#SBATCH -N 1               # Ensure that all cores are on one machine
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:4
#SBATCH --gpu-freq=high
#SBATCH -t 3-24:00:00
#SBATCH --mem=64G
#SBATCH --workdir=~/EasyMocap
#SBATCH -o ./logs/mvmp_train_%j.out 
#SBATCH -e ./logs/mvmp_train_%j.err 

# Load software modules and source conda environment
module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
source activate easymocap
nvidia-smi

# Run program
python3 apps/neuralbody/demo.py --mode soccer1_6 data/soccer1_6_with_annot --gpus 0,1,2,3