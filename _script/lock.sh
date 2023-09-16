#!/bin/bash  

# tmux new -As vc-backend2 "salloc -p gpu_test -t 0-01:00 --mem=8000 --gres=gpu:1"
salloc -c 8 -p gpu_test -t 0-01:00 --mem=64000 --gres=gpu:4