#!/bin/bash

for year in 2023; do
    sbatch --job-name=a$year /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/data-creation/africa/data-creation-africa.sh $year
done
