#!/bin/bash
#SBATCH  -J inference
#SBATCH  -t 02:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:a100:1
#SBATCH  --constraint=80g



eval "$(conda shell.bash hook)"
conda activate med-ehr
set -x



echo "top=1"

python inference.py 

echo "Job completed "
