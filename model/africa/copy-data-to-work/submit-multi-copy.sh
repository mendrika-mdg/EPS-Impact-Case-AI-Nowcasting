for LEAD_TIME in {0..6}; do 
    sbatch -J $LEAD_TIME /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/copy-data-to-work/copy-to-work-lead-time.sh $LEAD_TIME
done
