#!/bin/bash

#SBATCH --job-name=H0
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --qos=long
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH --output=/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/pc-X0/output/%j.out
#SBATCH --error=/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/pc-X0/error/%j.err

# Load environment
module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

# Check if zone_name is provided
if [ -z "$1" ]; then
  echo "Error: year argument is missing."
  echo "Usage: sbatch this_script.sh year"
  exit 1
fi

year="$1"

# Run the script
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/pc-X0/compute-pc-X0.py "$year"
