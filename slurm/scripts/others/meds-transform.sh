#!/bin/bash
#SBATCH  -J preprocess
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -q shamout
#SBATCH  -N 1
#SBATCH  -c 32
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err



# module load gcc

HYDRA_FULL_ERROR=1 MEDS_transform-pipeline transform.yaml



