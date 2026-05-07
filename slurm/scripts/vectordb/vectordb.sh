#!/bin/bash
#SBATCH  -J zest
#SBATCH  -t 10-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:a100:1




OVERLAY=path/to/
SIF=path/to/



singularity exec --nv --overlay "${OVERLAY}" "${SIF}" bash -lc "
  source path/to/conda.sh
  conda activate med-ehr
  set -x
  cd path/to/
  python create_vdb_idx.py
"

