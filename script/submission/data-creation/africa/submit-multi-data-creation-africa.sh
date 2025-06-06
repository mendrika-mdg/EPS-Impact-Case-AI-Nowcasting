#!/bin/bash

for year in 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023; do
    sbatch --job-name=a$year /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/data-creation/africa/data-creation-africa.sh $year
done
