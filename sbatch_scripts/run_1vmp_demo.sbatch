#!/bin/bash
#SBATCH -p seas_gpu
#SBATCH -N 1               # Ensure that all cores are on one machine
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:4
#SBATCH --gpu-freq=high
#SBATCH -t 0-02:00
#SBATCH --mem=64G
#SBATCH -o ./logs/1vmp_demo_%j.out 
#SBATCH -e ./logs/1vmp_demo_%j.err 

# Load software modules and source conda environment
module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
source activate easymocap
nvidia-smi

# Run program
# python3 apps/neuralbody/demo.py --mode soccer1_6_v_1 data/soccer1_6_with_annot --gpus 0,1,2,3 --demo
# python3 apps/neuralbody/demo.py --mode soccer1_6_v_1 data/soccer1_6_with_annot --gpus 0,1,2,3 --trainvis
python3 apps/neuralbody/demo.py --mode soccer1_6_v_1 data/soccer1_6_with_annot --gpus 0,1,2,3 --test