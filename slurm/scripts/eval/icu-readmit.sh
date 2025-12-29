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



# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/roberta.yaml
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/longformer.yaml 
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/big_bird.yaml 
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/roformer.yaml 
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/modernbert.yaml


# # descemb cls-ft
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/descemb.yaml --freeze

# # descemb -funetune
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/descemb.yaml

# genhpf
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/genhpf.yaml

# remed
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/remed.yaml



# med-bert
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/medbert.yaml

# cehrbert
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/cehrbert.yaml

# behrt
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/behrt.yaml

# hibehrt
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/hibehrt.yaml


# ehrmaba
# python final_eval.py --config-path /scratch/sas10092/ehr-foundation/slurm/config/eval/icu-readmit/ehrmamba.yaml