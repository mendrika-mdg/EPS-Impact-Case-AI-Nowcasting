#!/bin/bash

# Detect failed jobs from sacct

sacct -u mendrika --format=JobID,JobName,State,ExitCode -S $(date +%Y-%m-%d) | grep FAILED > failed_jobs.txt

while read -r line; do
    jobname=$(echo $line | awk '{print $2}')
    year=${jobname:1}  # assuming your jobs are named sYYYY
    echo "Resubmitting for year $year"
    sbatch --job-name=s$year /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/data-creation/africa/data-creation-africa.sh $year
    sleep 5
done < failed_jobs.txt
