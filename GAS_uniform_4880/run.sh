#!/bin/bash
#SBATCH -A yangzhijian
#SBATCH --gres=gpu:1
#SBATCH -J 4880_uni
#SBATCH -o output.log
#SBATCH -e err.log
#SBATCH -p gpu

/home/lidi/project/anaconda3/envs/pytorch_lastest/bin/python -u GAS.py