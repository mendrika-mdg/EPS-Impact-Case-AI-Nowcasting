#!/bin/bash

for i in {1..20}; do
    zone="zone_${i}"
    sbatch /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/H0-analysis/create-database.sh "$zone"
done
