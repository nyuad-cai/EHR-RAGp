#!/bin/bash
#SBATCH  -J medgemma_p
#SBATCH  -t 10-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -c 32
#SBATCH  -p nvidia
#SBATCH  -q shamout
#SBATCH  --gres=gpu:a100:1
#SBATCH  --constraint=80g
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err

#--gres=gpu:1
# -q nvidia-xxl
# -q shamout



module load gcc

srun python medgemma.py