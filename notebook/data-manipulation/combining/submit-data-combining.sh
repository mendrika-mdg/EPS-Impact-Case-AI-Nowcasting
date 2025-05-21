#!/bin/bash

#SBATCH --job-name="data-combination"
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/H0-analysis/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/H0-analysis/error/%j.err

# Load the required module
module load jaspy/3.11

# Activate the Python virtual environment
source /home/users/mendrika/SSA/bin/activate

# Executable
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/notebook/data-manipulation/combining/data-creation-before-t0-lead-time-for-map-more-lt.py