#!/bin/bash

#SBATCH --job-name=copy-to-work
#SBATCH --time=48:00:00
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=wiser-ewsa
#SBATCH --exclude=host1114
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/error/%j.err

set -e

# Define paths
GWS_BASE="/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa"
WORK_BASE="/work/scratch-nopw2/mrakotomanga/eps/pancast"

echo "Creating target directories..."
mkdir -p $WORK_BASE/inputs_t0
mkdir -p $WORK_BASE/targets_t1
mkdir -p $WORK_BASE/targets_t3
mkdir -p $WORK_BASE/targets_t6

echo "Copying inputs_t0 to $WORK_BASE/inputs_t0... at $(date)"
find $GWS_BASE/inputs_t0 -name "*.pt" -print0 | xargs -0 -n 100 cp -t $WORK_BASE/inputs_t0

echo "Copying targets_t1 to $WORK_BASE/targets_t1... at $(date)"
find $GWS_BASE/targets_t1 -name "*.pt" -print0 | xargs -0 -n 100 cp -t $WORK_BASE/targets_t1

echo "Copying targets_t3 to $WORK_BASE/targets_t3... at $(date)"
find $GWS_BASE/targets_t3 -name "*.pt" -print0 | xargs -0 -n 100 cp -t $WORK_BASE/targets_t3

echo "Copying targets_t6 to $WORK_BASE/targets_t6... at $(date)"
find $GWS_BASE/targets_t6 -name "*.pt" -print0 | xargs -0 -n 100 cp -t $WORK_BASE/targets_t6

echo "✅ All files copied successfully at $(date)"
