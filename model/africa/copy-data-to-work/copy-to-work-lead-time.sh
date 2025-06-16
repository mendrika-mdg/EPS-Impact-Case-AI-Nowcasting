#!/bin/bash

#SBATCH --job-name=copy-to-work
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=wiser-ewsa
#SBATCH --exclude=host1114
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/error/%j.err

set -e

# Input
LEAD_TIME=$1

# Define paths
GWS_BASE="/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa"
WORK_BASE="/work/scratch-nopw2/mrakotomanga/eps/pancast"

# Create target directory
mkdir -p $WORK_BASE/targets_t${LEAD_TIME}

echo "Copying targets_t${LEAD_TIME} to $WORK_BASE/targets_t${LEAD_TIME}... at $(date)"
find $GWS_BASE/targets_t${LEAD_TIME} -name "*.pt" -print0 | xargs -0 --no-run-if-empty -n 100 cp -t $WORK_BASE/targets_t${LEAD_TIME}

echo "✅ All files copied successfully at $(date)"
