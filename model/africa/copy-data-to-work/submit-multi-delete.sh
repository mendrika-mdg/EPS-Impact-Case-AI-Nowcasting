for LEAD_TIME in 2 4 5; do 
    sbatch -J "lead${LEAD_TIME}" /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/copy-data-to-work/delete.sh $LEAD_TIME
done
