#!/bin/bash
#SBATCH --job-name=debug-test
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --array=0-1
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/array/output/%A_%a.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/array/error/%A_%a.err

set -euxo pipefail

echo "Running SLURM task ID $SLURM_ARRAY_TASK_ID"
$hostname
$date
