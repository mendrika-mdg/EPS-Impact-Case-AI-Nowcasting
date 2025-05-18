#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/data-senegal/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/data-senegal/error/%j.err

# Load the required module
module load jaspy/3.11

# Activate the Python virtual environment
source /home/users/mendrika/SSA/bin/activate

year=$1

# Execute the Python script with proper argument passing
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/data-creation/senegal.py "$year"

# Check if the Python script executed successfully
if [ $? -ne 0 ]; then
    echo "Error: Python script did not execute successfully."
    exit 1
fi

echo "Job completed successfully."

