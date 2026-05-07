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

##SBATCH  --constraint=80g

##SBATCH -q cair

##SBATCH -q shamout

#SBATCH  -q nvidia-xxl




OVERLAY=/scratch/sas10092/ehr-foundation/overlay-512000M-15000K.ext3
SIF=/share/apps/admin/singularity-images/centos-8.2.2004.sif



singularity exec --nv --overlay "${OVERLAY}" "${SIF}" bash -lc "
  source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
  conda activate med-ehr
  set -x
  cd /scratch/sas10092/ehr-foundation
  python create_vdb_idx.py
"

