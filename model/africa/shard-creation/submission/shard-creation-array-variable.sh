#!/bin/bash

DESCRIPTION="shard-creation-t1-train"  # Replace this with a meaningful identifier

# Create log directories
mkdir -p /work/scratch-nopw2/mrakotomanga/eps/log/${DESCRIPTION}/output
mkdir -p /work/scratch-nopw2/mrakotomanga/eps/log/${DESCRIPTION}/error

#SBATCH --job-name=shard-creation
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --array=0-463  # Make sure to adjust this based on number of shards
#SBATCH --account=wiser-ewsa
#SBATCH --exclude=host1114
#SBATCH -o /work/scratch-nopw2/mrakotomanga/eps/log/shard-creation-t1-train/output/%A_%a.out
#SBATCH -e /work/scratch-nopw2/mrakotomanga/eps/log/shard-creation-t1-train/error/%A_%a.err

set -e
echo "Started on $(hostname) at $(date)"
echo "Running SLURM Array Task ID $SLURM_ARRAY_TASK_ID"

# Activate Python environment
source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

PARTITION="train"
LEAD_TIME=1

# Call the Python script
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/shard-creation/script/shard-creation-array-variable.py $PARTITION $LEAD_TIME

echo "Job completed at $(date)"
