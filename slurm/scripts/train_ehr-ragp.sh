#!/bin/bash
#SBATCH  -J ehr-ragp
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 8
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err

##SBATCH  --gres=gpu:h100:1

#SBATCH  --gres=gpu:h200:1
##SBATCH --constraint=80g

#SBATCH -q shamout
##SBATCH -q nvidia-xxl
##SBATCH -q cair



OVERLAY=/scratch/xxxx-1/ehr-foundation/overlay-512000M-15000K.ext3
SIF=/share/apps/admin/singularity-images/centos-8.2.2004.sif

# common settings 
BENCHMARK=mimic
VERSION=with-rtrieval
PROJECT_NAME=ehr-ragp
LOG_DIR=./models/ehr-ragp
TOKENIZER_PATH=./resources/vocab.json
DATA_PATH=./data/meds_normalized_arrow 
DATA_IDX_PATH=./resources/downstream_index.parquet
RUN_MODE=hparams

CHUNKING_STRATEGY=overlap
SPAN=256
USE_PROTOTYPES=0
UNIFORM=0

# # task
# TASK=y_icu_readmit_30
# MAIN_WINDOW_QUERY=within_stay_query
# MAIN_WINDOW_HISTORY=within_stay_hist_full

# TASK=y_mort
# MAIN_WINDOW_QUERY=within48_query
# MAIN_WINDOW_HISTORY=within48_hist_full

# TASK=y_los_7
# MAIN_WINDOW_QUERY=within24_query
# MAIN_WINDOW_HISTORY=within24_hist_full

TASK=y_mort_12mo
MAIN_WINDOW_QUERY=within_stay_query
MAIN_WINDOW_HISTORY=within_stay_hist_full


CKPT=./models/pretraining/wandb/run-20260810_093926-roberta_transformer_17152730_512_64_15_maskprob_12.5overlap/files/ckpt/roberta.ckpt
BACKBONE=roberta

# CKPT=./models/pretraining/wandb/run-20260810_092354-bert_medbert_17152697_512_64_15_maskprob_12.5overlap/files/ckpt/med-bert.ckpt
# BACKBONE=bert
# VARIANT=medbert

# CKPT=./models/pretraining/wandb/run-20260810_092628-bert_cehrbert_17152700_512_64_15_maskprob_12.5overlap/files/ckpt/cehrbert.ckpt
# BACKBONE=bert
# VARIANT=cehrbert

# CKPT=./models/pretraining/wandb/run-20260810_092912-bert_behrt_17152720_512_64_15_maskprob_12.5overlap/files/ckpt/behrt.ckpt
# BACKBONE=bert
# VARIANT=behrt

# CKPT=./models/pretraining/wandb/run-20260810_093214-bert_hibehrt_17152724_512_64_15_maskprob_12.5overlap/files/ckpt/hibehrt.ckpt
# BACKBONE=bert
# VARIANT=hibehrt


SEQ_LENGTH_Q=512
OVERLAP_Q=0
singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" bash -lc "
  source /share/apps/xxxx-3/miniconda/3-4.11.0/etc/profile.d/conda.sh
  conda activate ehr-ragp
  set -x
  umask 0002
  cd /scratch/xxxx-1/ehr-foundation
  torchrun --master_port=$((20000 + (SLURM_JOB_ID % 20000))) --nproc_per_node=1 train_ehr_ragp.py \
    --backbone-name ${BACKBONE} \
    --job-id ${SLURM_JOB_ID} \
    --version ${VERSION} \
    --wandb-api-key $(cat ~/.wandb_token) \
    --wandb-log-dir ${LOG_DIR} \
    --task ${TASK} \
    --data-idx-path ${DATA_IDX_PATH} \
    --data-path ${DATA_PATH} \
    --tokenizer-path ${TOKENIZER_PATH} \
    --seq-length-q ${SEQ_LENGTH_Q} \
    --overlap-q ${OVERLAP_Q} \
    --main-window-query ${MAIN_WINDOW_QUERY} \
    --main-window-history ${MAIN_WINDOW_HISTORY} \
    --ckpt-path ${CKPT} \
    --chunking-strategy ${CHUNKING_STRATEGY} \
    --benchmark ${BENCHMARK} \
    --span ${SPAN} \
    --project-name ${PROJECT_NAME}  \
    $( [ -n "${VARIANT:-}" ] && echo "--variant ${VARIANT}" ) \
    $( [ "$USE_PROTOTYPES" -eq 1 ] && echo "--use-prototypes" ) \
    $( [ "$UNIFORM" -eq 1 ] && echo "--uniform" )
"



#####################################################################
# CLMBR training script
#####################################################################







# # common settings 
# BENCHMARK=ehrshot
# BACKBONE=clmbr 
# VERSION=with-rtrieval
# PROJECT_NAME=ehr-ragp
# LOG_DIR=./models/clmbr
# TOKENIZER_PATH=${SCRATCH}/.cache/huggingface/hub/models--StanfordShahLab--clmbr-t-base/snapshots/c7a5f4db6089525e374c2c90372350db3a043d73/clmbr_v8_original_dictionary.json
# DATA_PATH=./data/ehrshot_clmbr_tokenized_arrow
# DATA_IDX_PATH=./resources/clmbr_idx.parquet
# RUN_MODE=hparams

# CHUNKING_STRATEGY=overlap
# SPAN=256
# USE_PROTOTYPES=1


# SEQ_LENGTH_Q=512
# OVERLAP_Q=0



# TASK=new_acutemi
# # TASK=new_celiac
# # TASK=new_hyperlipidemia
# # TASK=new_hypertension
# # TASK=new_lupus
# # TASK=new_pancan

# # TASK=guo_icu
# # TASK=guo_los
# # TASK=guo_readmission


# torchrun --master_port=$((20000 + (SLURM_JOB_ID % 20000))) --nproc_per_node=1 train_ehr_ragp.py \
#     --backbone-name ${BACKBONE} \
#     --job-id ${SLURM_JOB_ID} \
#     --version ${VERSION} \
#     --wandb-api-key $(cat ~/.wandb_token) \
#     --wandb-log-dir ${LOG_DIR} \
#     --project-name ${PROJECT_NAME} \
#     --task ${TASK} \
#     --data-path ${DATA_PATH} \
#     --data-idx-path ${DATA_IDX_PATH} \
#     --tokenizer-path ${TOKENIZER_PATH} \
#     --seq-length-q ${SEQ_LENGTH_Q} \
#     --overlap-q ${OVERLAP_Q} \
#     --chunking-strategy ${CHUNKING_STRATEGY} \
#     --span ${SPAN} \
#     --benchmark ${BENCHMARK} \
#     $( [ "$USE_PROTOTYPES" -eq 1 ] && echo "--use-prototypes" )
  
