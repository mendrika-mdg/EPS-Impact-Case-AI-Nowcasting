#!/bin/bash

#SBATCH --job-name=shard-creation
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --array=0-463
#SBATCH --account=wiser-ewsa
#SBATCH --exclude=host1114
#SBATCH -o /work/scratch-nopw2/mrakotomanga/eps/log/output/%A_%a.out
#SBATCH -e /work/scratch-nopw2/mrakotomanga/eps/log/error/%A_%a.err

set -e

echo "Started on $(hostname) at $(date)"
echo "Running SLURM Array Task ID $SLURM_ARRAY_TASK_ID"

source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

# Optional: create output directory if using scratch
mkdir -p /work/scratch-nopw2/mrakotomanga/eps/shards/t1/val

python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/shard-creation/script/shard-creation-array.py

echo "Job completed at $(date)"
