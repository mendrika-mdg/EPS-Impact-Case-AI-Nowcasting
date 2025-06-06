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

set -euo pipefail

module load jaspy/3.11

source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

YEAR=$1

python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/data-creation/africa-mpi.py "${YEAR}"

echo "Job completed successfully."
