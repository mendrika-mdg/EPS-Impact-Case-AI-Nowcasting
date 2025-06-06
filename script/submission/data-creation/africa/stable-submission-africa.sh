#!/bin/bash

#SBATCH --job-name=africa-creation
#SBATCH --time=48:00:00
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/data-creation-africa/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/data-creation-africa/error/%j.err

set -e

# Activate your environment
source /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/.venv/bin/activate

# Attempt GWS mount with retries
for i in {1..3}; do
    gwsmount /gws/nopw/j04/wiser_ewsa/mrakotomanga && break
    echo "Retrying gwsmount ($i)..."
    sleep 5
done

# Test if GWS is really accessible
if ! ls /gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Africa > /dev/null 2>&1; then
    echo "GWS mounted but inaccessible. Exiting safely."
    exit 1
fi

# Short pause before starting
sleep 5

# Launch python
python /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/data-creation/africa-mpi.py

echo "Job completed successfully."
