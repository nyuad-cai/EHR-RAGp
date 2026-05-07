#!/bin/bash
#SBATCH  -J zllm
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:h200:1

##SBATCH  --constraint=80g

##SBATCH -q cair

##SBATCH -q shamout

#SBATCH  -q nvidia-xxl

python llm_eval.py --model-name epfl-llm/meditron-7b

