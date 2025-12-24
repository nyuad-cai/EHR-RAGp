#!/bin/bash
#SBATCH  -J icu-readmit
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 32
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:a100:1

##SBATCH -q shamout

##SBATCH -q nvidia-xxl

##SBATCH -q cair


##SBATCH  --constraint=80g


eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x


# done # python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/bert.yaml
# done # python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/roberta.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/longformer.yaml # continues
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/big_bird.yaml # continues
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/roformer.yaml # continues
# done # python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/modernbert.yaml


# # descemb cls-ft
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/descemb.yaml --freeze

# # descemb -funetune
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/descemb.yaml

# genhpf
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/genhpf.yaml

# remed
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/icu-readmit/remed.yaml