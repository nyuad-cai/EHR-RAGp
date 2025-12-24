#!/bin/bash
#SBATCH  -J long-los
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 32
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:a100:1

##SBATCH -q nvidia-xxl
##SBATCH -q shamout
##SBATCH -q cair


##SBATCH  --constraint=80g




eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x



# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/long-los/roberta.yaml
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/long-los/longformer.yaml 
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/long-los/big_bird.yaml 
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/long-los/roformer.yaml
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/long-los/modernbert.yaml 