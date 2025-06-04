#!/bin/bash

#SBATCH --job-name=africa-nbx0
#SBATCH --time=48:00:00
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/error/%j.err

set -e

module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/x0-analysis/nb-x0-panafrica-mpi.py

echo "Job completed successfully."
