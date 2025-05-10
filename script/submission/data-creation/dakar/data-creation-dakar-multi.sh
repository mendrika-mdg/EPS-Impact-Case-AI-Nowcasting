#!/bin/bash

for year in {2004..2023}; do
    sbatch /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/data-creation/dakar/data-creation-dakar.sh $year
done
