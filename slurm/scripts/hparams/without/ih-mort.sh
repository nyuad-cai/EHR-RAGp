#!/bin/bash
#SBATCH  -J ih-mort
#SBATCH  -t 4-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 32
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:h200:1

##SBATCH -q nvidia-xxl
#SBATCH -q shamout
##SBATCH -q cair


##SBATCH  --constraint=80g


eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x


# done # python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/bert.yaml
# done # python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/roberta.yaml
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/longformer.yaml # continues
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/big_bird.yaml # continues
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/roformer.yaml # continues
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/modernbert.yaml
python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/modernbert-long.yaml



# # descemb cls-ft
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/descemb.yaml --freeze

# # descemb -funetune
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/descemb.yaml

# genhpf
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/genhpf.yaml

# remed
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/remed.yaml




# med-bert
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/medbert.yaml

# cehrbert
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/cehrbert.yaml

# behrt
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/behrt.yaml

# hibehrt
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/hibehrt.yaml


# ehrmaba
# python wo_hparams_opt.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/hparams/ih-mort/ehrmamba.yaml