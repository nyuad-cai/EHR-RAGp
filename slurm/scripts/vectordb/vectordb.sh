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


#SBATCH  -q shamout
#SBATCH  --constraint=80g
##SBATCH  -q nvidia-xxl

#--gres=gpu:1
# -q nvidia-xxl
# -q shamout
# -p nvidia
# -q shamout
# --gres=gpu:a100:1
# --constraint=80g




eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x


python vectordb.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/vectordb/config.yaml