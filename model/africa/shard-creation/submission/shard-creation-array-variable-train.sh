#!/bin/bash

#SBATCH --job-name=train-3
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --array=0-463                                           # Make sure to adjust this based on number of shards
#SBATCH --account=wiser-ewsa
#SBATCH --exclude=host1114
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/array/output/%A_%a.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/array/error/%A_%a.err


set -e
echo "Started on $(hostname) at $(date)"
echo "Running SLURM Array Task ID $SLURM_ARRAY_TASK_ID"

# Activate Python environment
source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

PARTITION="train"
LEAD_TIME=3

# Call the Python script
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/shard-creation/script/shard-creation-array-variable.py $PARTITION $LEAD_TIME

echo "Job completed at $(date)"
