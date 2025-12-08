#!/bin/bash
#SBATCH  -J test
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  --gres=gpu:a100:1
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
export DATA_PATH="/scratch/sas10092/ehr-foundation/data/meds_normalized_arrow"
export DATA_IDX_PATH="/scratch/sas10092/ehr-foundation/notebooks/mortality_labels.parquet"
export WANDB_API_KEY="59b6438e0496b3089f91abef35d31dae69b6c009"
export BACKBONE="Roformer_base"
export LOG_DIR="./models/eval/"
export MAIN_WINDOW="within48_query"




# export VERSION="fine-tune-random"
# 1536
# export CKPT_PATH="/scratch/sas10092/ehr-foundation/models/mlm/wandb/run-20250925_120754-Roformer_base_12116792_1536_192/files/ckpt/epoch=23-step=407928.ckpt"
# 1024
# export CKPT_PATH="/scratch/sas10092/ehr-foundation/models/mlm/wandb/run-20250925_120956-Roformer_base_12116794_1024_128/files/ckpt/epoch=24-step=289725.ckpt"
# 512
# export CKPT_PATH="/scratch/sas10092/ehr-foundation/models/mlm/wandb/run-20250925_121053-Roformer_base_12116797_512_64/files/ckpt/epoch=18-step=199538.ckpt"





# export VERSION="from-scratch"

eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x


python mlm_eval.py \
    --learning-rate 1e-6 \
    --weight-decay 1e-4 \
    --max-epochs 50 \
    --batch-size 16 \
    --chunk-length 1536 \
    --overlap 192 \
    --pretrained \
    # --freeze 

