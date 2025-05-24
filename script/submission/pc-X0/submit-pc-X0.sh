# Loop through years 2020 to 2023
for year in {2020..2023}; do
    sbatch --job-name="${year}" /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/pc-X0/pc-X0.sh "${year}"
done

