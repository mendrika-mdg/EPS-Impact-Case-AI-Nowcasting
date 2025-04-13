#!/bin/bash

#SBATCH --job-name="find-nbx0"
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/error/%j.err

# Load the required module
module load jaspy/3.11

# Activate the Python virtual environment
source /home/users/mendrika/SSA/bin/activate

# Parse arguments
domain_lat_min=$1
domain_lat_max=$2
domain_lon_min=$3
domain_lon_max=$4
region_name=$5
lead_time=$6

# Execute the Python script with proper argument passing
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/x0-analysis/nb-x0.py \
  "$domain_lat_min" "$domain_lat_max" "$domain_lon_min" "$domain_lon_max" "$region_name" "$lead_time"

# Check if the Python script executed successfully
if [ $? -ne 0 ]; then
    echo "Error: Python script did not execute successfully."
    exit 1
fi

echo "Job completed successfully."
