#!/bin/bash

# Array of hours
hours=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23")


for hour in "${hours[@]}"; do
  sbatch -J "$hour" /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/pc/compute-pc.sh "$hour"
done

