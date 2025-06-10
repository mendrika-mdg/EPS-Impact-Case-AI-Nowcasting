#!/bin/bash

#SBATCH --job-name=africa-creation
#SBATCH --time=48:00:00
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/error/%j.err

set -e  # Exit on error

echo "Started on $(hostname) at $(date)"


echo "Activating virtual environment..."
source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

# Slight pause to allow GWS to settle
sleep 5

echo "Launching job script..."
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/shard-creation.py

echo "Job completed at $(date)"
