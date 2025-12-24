#!/bin/bash
#SBATCH  -J simclr
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia

#SBATCH  -c 32
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:v100:2


##SBATCH  --constraint=80g
##SBATCH -q shamout
##SBATCH -q cair
##SBATCH -q nvidia-xxl






export TOKENIZER_PATH="/scratch/sas10092/ehr-foundation/vocab.json"
#export DATA_PATH="/scratch/sas10092/ehr-foundation/data/meds_normalized_arrow"
export DATA_PATH="/scratch/sas10092/ehr-foundation/data/desc_gen_dataset/"
#export DATA_IDX_PATH="/scratch/sas10092/ehr-foundation/pretrain_idx.parquet"
export DATA_IDX_PATH="/scratch/sas10092/ehr-foundation/downstream_idx.parquet"
export WANDB_API_KEY="59b6438e0496b3089f91abef35d31dae69b6c009"
#export LOG_DIR="/scratch/sas10092/ehr-foundation/models/mlm"
export LOG_DIR="/scratch/sas10092/ehr-foundation/models/simclr"
export VERSION="15_maskprob_12_5overlap"
export PRETRAIN_MODE="simclr"
export BASELINE="genhpf"

# "15_maskprob_12_5overlap"
eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x


export BACKBONE="genhpf_simclr"
torchrun --nproc_per_node=2 pretrain.py \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --max-epochs 100 \
    --batch-size 64 \
    --chunk-length 511 \
    --overlap 0 


# export BACKBONE="bert"
# torchrun --nproc_per_node=2 pretrain.py \
#     --learning-rate 5e-5 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 128 \
#     --chunk-length 512 \
#     --overlap 64 

# export BACKBONE="mamba"
# torchrun --nproc_per_node=4 pretrain.py \
#     --learning-rate  \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 8 \
#     --chunk-length 2048 \
#     --overlap 0




# export BACKBONE="roberta"
# torchrun --nproc_per_node=4 mlm_pretrain.py \
#     --learning-rate 2.2908676527677725e-05 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 128 \
#     --chunk-length 512 \
#     --overlap 64




# export BACKBONE="longformer"
# torchrun --nproc_per_node=4 mlm_pretrain.py \
#     --learning-rate 2.2908676527677725e-05 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 32 \
#     --chunk-length 1024 \
#     --overlap 128



# export BACKBONE="big_bird"
# torchrun --nproc_per_node=4 mlm_pretrain.py \
#     --learning-rate 2.2908676527677725e-05 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 32 \
#     --chunk-length 1024 \
#     --overlap 128



# export BACKBONE="modernbert"
# torchrun --nproc_per_node=4 mlm_pretrain.py \
#     --learning-rate 2.2908676527677725e-05 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 32 \
#     --chunk-length 1024 \
#     --overlap 28


# export BACKBONE="roformer"
# torchrun --nproc_per_node=4 mlm_pretrain.py \
#     --learning-rate 2.2908676527677725e-05 \
#     --weight-decay 1e-2 \
#     --max-epochs 100 \
#     --batch-size 16 \
#     --chunk-length 1024 \
#     --overlap 128