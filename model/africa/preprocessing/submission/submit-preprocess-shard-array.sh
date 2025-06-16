#!/bin/bash

#SBATCH --job-name=shard-creation
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --array=0-451  
#SBATCH --account=wiser-ewsa
#SBATCH --exclude=host1114
#SBATCH -o /work/scratch-nopw2/mrakotomanga/eps/log/output/%A_%a.out
#SBATCH -e /work/scratch-nopw2/mrakotomanga/eps/log/error/%A_%a.err

set -e

echo "Started on $(hostname) at $(date)"
echo "Running SLURM Array Task ID $SLURM_ARRAY_TASK_ID"

source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

PARTITION="train"
LEAD_TIME=1

python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/preprocessing/script/preprocess-shard-array.py $PARTITION $LEAD_TIME

echo "Job completed at $(date)"