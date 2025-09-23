#!/bin/bash
#SBATCH  -J test
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -q shamout
#SBATCH  -p nvidia
#SBATCH  --gres=gpu:a100:4
#SBATCH  --constraint=80g
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




export DATA_PATH="/scratch/sas10092/ehr-foundation/data/meds_normalized/data/train"
export DATA_IDX_PATH="/scratch/sas10092/ehr-foundation/dataset_idx.parquet"
export TOKENIZER_PATH="/scratch/sas10092/ehr-foundation/vocab.json"
export WANDB_API_KEY=""
export LOG_DIR="./models/pretraining/"
export VERSION=""
export RUN_NAME=""

eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x

srun python mlm_pretrain.py