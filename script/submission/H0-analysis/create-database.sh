#!/bin/bash

#SBATCH --job-name=H0
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH --output=/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/H0-analysis/output/%j.out
#SBATCH --error=/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/H0-analysis/error/%j.err

# Load environment
module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

# Check if zone_name is provided
if [ -z "$1" ]; then
  echo "Error: zone_name argument is missing."
  echo "Usage: sbatch this_script.sh zone_1"
  exit 1
fi

zone_name="$1"

# Run the script
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/H0-analysis/create-database.py "$zone_name"
