#!/bin/bash
#SBATCH  -J preprocess
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -c 32
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err

#--gres=gpu:1
# -q nvidia-xxl
# -q shamout

# -p nvidia
# -q shamout
# --gres=gpu:a100:1
# --constraint=80g

# module load gcc

srun python preprocess.py
