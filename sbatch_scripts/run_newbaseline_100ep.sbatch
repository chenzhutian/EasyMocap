#!/bin/bash
#SBATCH -p seas_gpu
#SBATCH -c 8
#SBATCH --gres=gpu:8
#SBATCH --gpu-freq=high
#SBATCH --constraint=cc7.0
#SBATCH -t 3-24:00:00
#SBATCH --mem=64G
#SBATCH -o newbaseline_100ep_%j.out 
#SBATCH -e newbaseline_100ep_%j.err 

# Load software modules and source conda environment
module load GCC/8.2.0-2.31.1
module load cuda/10.2.89-fasrc01 
module load cudnn/7.6.5.32_cuda10.2-fasrc01
source activate py1.10_cuda102
nvidia-smi

# Run program
srun -c 8 --gres=gpu:8 bash train_newbaseline_100ep.sh