#!/bin/bash

for year in {2004..2024}; do
    sbatch --job-name=s$year /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/data-creation/zambia/data-creation-zambia.sh $year
done
