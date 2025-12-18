#!/bin/bash
#SBATCH  -J 1y-mort
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 32
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:a100:1

##SBATCH -q cair

#SBATCH -q shamout

##SBATCH  -q nvidia-xxl
##SBATCH  --constraint=80g

eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x

# done # python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/bert.yaml
# done # python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/roberta.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/longformer.yaml # continues
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/big_bird.yaml # continues
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/roformer.yaml # continues
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/modernbert.yaml # continues