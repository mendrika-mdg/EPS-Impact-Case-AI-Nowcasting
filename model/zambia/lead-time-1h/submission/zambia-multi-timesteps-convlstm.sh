#!/bin/bash
#SBATCH --job-name=zambia-nowcast
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # One task per GPU
#SBATCH --gres=gpu:4                # 4 GPUs per node
#SBATCH --cpus-per-task=4           # Tune based on data loading needs
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpuhost006        # Avoid node with broken GPUs
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/deep-learning/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/deep-learning/error/%j.err

# Load your environment
module load jaspy/3.11
source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

# Log GPU/node info for debugging
echo "Running on node: $(hostname)"
nvidia-smi

# IMPORTANT: use `srun` to launch PyTorch Lightning with DDP
srun python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/zambia/lead-time-1h/Zambia-multi-timesteps-t1-convlstm.py
