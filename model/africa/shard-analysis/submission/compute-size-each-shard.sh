#!/bin/bash

#SBATCH --job-name=size-train
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=wiser-ewsa
#SBATCH --exclude=host1114
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/error/%j.err

set -e  # Exit immediately on error

echo "Started on $(hostname) at $(date)"

# === Activate virtual environment ===
echo "Activating virtual environment..."
source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

# === Optional short pause ===
sleep 5

PARTITION="train"
LEAD_TIME=1

# === Launch Python script ===
echo "Shard size computation script..."
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/shard-analysis/script/compute-size-each-shard.py $PARTITION $LEAD_TIME

echo "Job completed at $(date)"
