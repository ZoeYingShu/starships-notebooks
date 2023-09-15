#!/bin/bash
#SBATCH --account=def-dlafre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=140G
#SBATCH --time=5-12:00
#SBATCH --job-name HRR_spirou
#SBATCH --output=/home/adb/scratch/sbatch_outputs/out_sbatch_%j.txt
#SBATCH --mail-type=FAIL

source ~/.venvs/starships_env_39/bin/activate

# !!!!!!! Don't forget to remove #SBATCH --array=1-10  !!!!!!!!!
echo "Starting python code..."
python retrieval_WASP-33b_JR_guillot_spirou_only.py

## Burnin (use with >>> #SBATCH --array=1-10 for example)
#echo "Starting python code with SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
#python retrieval_WASP-33b_JR_guillot_spirou_only.py idx_file=$SLURM_ARRAY_TASK_ID
