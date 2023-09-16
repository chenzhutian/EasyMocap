#!/bin/bash  

module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01

source activate easymocap

# ssh -N -f -R 9902:localhost:9902 $HOSTNAME
# ssh -N -f -R 22022:localhost:22022 $HOSTNAME