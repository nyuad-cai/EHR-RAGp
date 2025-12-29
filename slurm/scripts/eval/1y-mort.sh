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

##SBATCH -q shamout

##SBATCH  -q nvidia-xxl
##SBATCH  --constraint=80g

eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x


# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/roberta.yaml
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/longformer.yaml 
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/big_bird.yaml 
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/roformer.yaml 
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/modernbert.yaml 

# # descemb cls-ft
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/descemb.yaml --freeze

# # descemb -funetune
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/descemb.yaml

# genhpf
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/genhpf.yaml

# remed
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/remed.yaml




# hibehrt
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/hibehrt.yaml

# medbert
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/medbert.yaml

# behrt
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/behrt.yaml

# cehrbert
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/cehrbert.yaml


# ehrmaba
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/1y-mort/ehrmamba.yaml