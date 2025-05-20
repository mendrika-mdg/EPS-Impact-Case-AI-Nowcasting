#!/bin/bash

#SBATCH --job-name="pc"
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/pc/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/pc/error/%j.err

# Load the required module
module load jaspy/3.11

# Activate the Python virtual environment
source /home/users/mendrika/SSA/bin/activate

hour="$1"

# Executable
module load "jaspy/3.11"
source /home/users/mendrika/SSA/bin/activate
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/pc/compute-pc.py "$hour"