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



##SBATCH  --constraint=80g


eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x


# python wo_hparams_opt.py --config-path path/to/bert.yaml
# python wo_hparams_opt.py --config-path path/to/roberta.yaml
# python wo_hparams_opt.py --config-path path/to/longformer.yaml 
# python wo_hparams_opt.py --config-path path/to/big_bird.yaml 
# python wo_hparams_opt.py --config-path path/to/roformer.yaml 
# python wo_hparams_opt.py --config-path path/to/modernbert.yaml 
python wo_hparams_opt.py --config-path path/to/modernbert-long.yaml 

# # descemb cls-ft
# python wo_hparams_opt.py --config-path path/to/descemb.yaml --freeze

# # descemb -funetune
# python wo_hparams_opt.py --config-path path/to/descemb.yaml

# genhpf
# python wo_hparams_opt.py --config-path path/to/genhpf.yaml



# hibehrt
# python wo_hparams_opt.py --config-path path/to/hibehrt.yaml

# medbert
# python wo_hparams_opt.py --config-path path/to/medbert.yaml

# behrt
# python wo_hparams_opt.py --config-path path/to/behrt.yaml

# cehrbert
# python wo_hparams_opt.py --config-path path/to/cehrbert.yaml

# remed
# python wo_hparams_opt.py --config-path path/to/remed.yaml


# ehrmaba
# python wo_hparams_opt.py --config-path path/to/ehrmamba.yaml