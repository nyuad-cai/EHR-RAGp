#!/bin/bash
#SBATCH  -J vectordb
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  --gres=gpu:a100:1
#SBATCH  -c 32
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err






eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x


python vectordb.py --config-path /path/to
