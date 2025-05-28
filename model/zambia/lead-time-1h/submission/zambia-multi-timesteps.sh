#!/bin/bash

#SBATCH --job-name=H0
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --qos=orchid
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/deep-learning/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/deep-learning/error/%j.err

# Load environment
module load jaspy/3.11
source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

# Run the script
srun python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/lead-time-1h/Zambia-multi-timesteps-t1.py
