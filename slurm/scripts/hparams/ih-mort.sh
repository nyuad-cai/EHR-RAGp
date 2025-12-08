#!/bin/bash
#SBATCH  -J ih-mort
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


# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/bert.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/roberta.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/longformer.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/big_bird.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/roformer.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/modernbert.yaml