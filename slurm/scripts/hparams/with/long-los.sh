#!/bin/bash
#SBATCH  -J long-los
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err

#SBATCH  --gres=gpu:h200:1

##SBATCH  --gres=gpu:h100:1

##SBATCH  --gres=gpu:a100:1
##SBATCH --constraint=80g

#SBATCH -q shamout
##SBATCH -q nvidia-xxl
##SBATCH -q cair



OVERLAY=/scratch/sas10092/ehr-foundation/overlay-512000M-15000K.ext3
SIF=/share/apps/admin/singularity-images/centos-8.2.2004.sif

CHUNKING_STRATEGY=overlap
SPAN=256
USE_PROTOTYPES=0

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" bash -lc "
  source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
  conda activate med-ehr
  set -x
  cd /scratch/sas10092/ehr-foundation
  torchrun --master_port=$((20000 + (SLURM_JOB_ID % 20000))) --nproc_per_node=1 w_hparams_opt.py \
  --config-path /scratch/sas10092/ehr-foundation/slurm/config/with_retrieval/hparams/long-los/roformer.yaml \
  --chunking-strategy "${CHUNKING_STRATEGY}" \
  --benchmark mimic \
  --span "${SPAN}" \
  $( [ "$USE_PROTOTYPES" -eq 1 ] && echo "--use-prototypes" )
"


