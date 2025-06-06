#!/bin/bash

#SBATCH --job-name=africa-creation
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/data-creation-africa/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/data-creation-africa/error/%j.err

# Fail immediately if any command exits with non-zero status
set -e

# Load the required module
module load jaspy/3.11

# Activate the Python virtual environment
source /home/users/mendrika/SSA/bin/activate

# Execute the Python script
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/x0-analysis/nb-x0-panafrica.py

echo "Job completed successfully."
