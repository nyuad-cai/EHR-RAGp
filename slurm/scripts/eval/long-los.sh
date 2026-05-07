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





eval "$(conda shell.bash hook)"
conda activate med-ehr

set -x



# python final_eval.py --config-path path/to/roberta.yaml
# python final_eval.py --config-path path/to/longformer.yaml 
# python final_eval.py --config-path path/to/big_bird.yaml 
python final_eval.py --config-path path/to/roformer.yaml 
# python final_eval.py --config-path path/to/modernbert.yaml 

# # descemb cls-ft
# python final_eval.py --config-path path/to/descemb.yaml --freeze

# # descemb -funetune
# python final_eval.py --config-path path/to/descemb.yaml

# genhpf
# python final_eval.py --config-path path/to/genhpf.yaml

# remed
# python final_eval.py --config-path path/to/remed.yaml


# hibehrt
# python final_eval.py --config-path path/to/hibehrt.yaml

# medbert
# python final_eval.py --config-path path/to/medbert.yaml

# behrt
# python final_eval.py --config-path path/to/behrt.yaml

# cehrbert
# python final_eval.py --config-path path/to/cehrbert.yaml


# ehrmaba
# python final_eval.py --config-path path/to/ehrmamba.yaml