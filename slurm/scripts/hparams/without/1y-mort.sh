#!/bin/bash
#SBATCH  -J 1y-mort
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:h100:1

#SBATCH -q cair

##SBATCH -q shamout

##SBATCH  -q nvidia-xxl
##SBATCH  --constraint=80g

eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x

# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/bert.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/roberta.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/longformer.yaml 
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/big_bird.yaml 
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/roformer.yaml 
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/modernbert.yaml 
python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/modernbert-long.yaml 

# # descemb cls-ft
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/descemb.yaml --freeze

# # descemb -funetune
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/descemb.yaml

# genhpf
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/genhpf.yaml



# hibehrt
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/hibehrt.yaml

# medbert
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/medbert.yaml

# behrt
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/behrt.yaml

# cehrbert
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/cehrbert.yaml

# remed
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/remed.yaml


# ehrmaba
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/1y-mort/ehrmamba.yaml

