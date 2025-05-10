#!/bin/bash

for year in {2004..2023}; do
    sbatch --job-name=s$year /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/data-creation/senegal/data-creation-senegal.sh $year
done
