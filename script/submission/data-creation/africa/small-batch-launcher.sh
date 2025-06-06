#!/bin/bash

# safer batch submitter

years=(2005 2007 2008 2011 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023)

batch_size=5
counter=0

for year in "${years[@]}"; do
    sbatch --job-name=s$year /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/data-creation/africa/data-creation-africa.sh $year
    echo "Submitted year $year"
    ((counter++))
    
    if (( counter % batch_size == 0 )); then
        echo "Waiting 60s to avoid overloading scheduler"
        sleep 60
    fi
done
