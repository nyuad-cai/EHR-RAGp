#!/bin/bash
#SBATCH  -J ih-mort
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err



#SBATCH  --gres=gpu:h100:1





OVERLAY=
SIF=

CHUNKING_STRATEGY=visit
SPAN=256
USE_PROTOTYPES=1

singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" bash -lc "
  source path/to/conda.sh
  conda activate med-ehr
  set -x
  cd path/to/
  torchrun --master_port=$((20000 + (SLURM_JOB_ID % 20000))) --nproc_per_node=1 w_hparams_opt.py \
  --config-path path/to/ \
  --chunking-strategy "${CHUNKING_STRATEGY}" \
  --span "${SPAN}" \
  $( [ "$USE_PROTOTYPES" -eq 1 ] && echo "--use-prototypes" )
"



