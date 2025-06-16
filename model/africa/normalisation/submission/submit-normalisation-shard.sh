#!/bin/bash

#SBATCH --job-name=shard-creation
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

# === Launch Python script ===
echo "Launching normalising parameters' computation script..."
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/normalisation/script/normalisation-param-shard.py

echo "Job completed at $(date)"
