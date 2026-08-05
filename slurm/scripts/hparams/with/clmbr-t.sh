#!/bin/bash
#SBATCH  -J clmbr
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err

##SBATCH  --gres=gpu:h200:1
#SBATCH  --gres=gpu:h100:1

##SBATCH  --gres=gpu:a100:1
##SBATCH --constraint=80g

#SBATCH -q shamout
##SBATCH -q nvidia-xxl
##SBATCH -q cair


CHUNKING_STRATEGY=overlap
SPAN=256
USE_PROTOTYPES=1

# new_acutemi
# new_celiac
# new_hyperlipidemia
# new_hypertension
# new_lupus
# new_pancan

# guo_icu
# guo_los
# guo_readmission

# chexpert
export TASK='guo_readmission'

eval "$(conda shell.bash hook)"
conda activate med-ehr
set -x
torchrun --master_port=$((20000 + (SLURM_JOB_ID % 20000))) --nproc_per_node=1 w_hparams_opt.py \
--config-path /scratch/sas10092/ehr-foundation/slurm/config/with_retrieval/hparams/clmbr.yaml \
--chunking-strategy "${CHUNKING_STRATEGY}" \
--span "${SPAN}" \
--benchmark ehrshot \
$([ "$USE_PROTOTYPES" -eq 1 ] && echo "--use-prototypes" )
