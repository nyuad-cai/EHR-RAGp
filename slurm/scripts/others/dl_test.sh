#!/bin/bash
#SBATCH  -J test
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -q shamout
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



export TOKENIZER_PATH="/scratch/sas10092/ehr-foundation/vocab.json"
export DATA_PATH="/scratch/sas10092/ehr-foundation/data/meds_normalized/data/train"
export DATA_IDX_PATH="/scratch/sas10092/ehr-foundation/dataset_idx.parquet"
export WANDB_API_KEY="59b6438e0496b3089f91abef35d31dae69b6c009"
export BACKBONE="Roformer_base"
export LOG_DIR="./models/mlm/"
export VERSION="pretrained"




eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x


torchrun --nproc_per_node=4 mlm_pretrain.py \
    --learning-rate 1e-5 \
    --weight-decay 1e-2 \
    --max-epochs 100 \
    --batch-size 16 \
    --chunk-length 1024 \
    --overlap 128 \
    --pretrained 