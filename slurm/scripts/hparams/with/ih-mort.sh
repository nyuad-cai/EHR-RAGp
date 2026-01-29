#!/bin/bash
#SBATCH  -J ih-mort
#SBATCH  -t 10-00:00:00
#SBATCH  -n 1
#SBATCH  -N 1
#SBATCH  -p nvidia
#SBATCH  -c 16
#SBATCH  -o ./slurm/logs/%x.%J.out
#SBATCH  -e ./slurm/logs/%x.%J.err
#SBATCH  --gres=gpu:a100:1
#SBATCH  --constraint=80g

#SBATCH -q shamout
##SBATCH  -q nvidia-xxl
##SBATCH -q cair



eval "$(conda shell.bash hook)"
conda activate med-ehr
set -x
ORIG_DB=/scratch/sas10092/ehr-foundation/data/vectordbs/roformer/mean/within48_hist_full_1024_128
LOCAL_DB=/tmpdata/chroma_${SLURM_JOB_ID}

echo "Copying Chroma DB to local /tmp..."
rsync -a --delete "${ORIG_DB}/" "${LOCAL_DB}/"

echo "Copying Completed!"

chmod -R u+rwx "${LOCAL_DB}"

export CHROMA_DB_PATH="${LOCAL_DB}"

# torchrun --nproc_per_node=2  
python w_hparams_opt.py \
    --config-path /scratch/sas10092/ehr-foundation/slurm/config/with_retrieval/hparams/ih-mort/roformer.yaml \
    --chroma-db-path "${CHROMA_DB_PATH}"

echo "Job completed"


