#!/bin/bash
#SBATCH --account=def-dlafre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=140G
#SBATCH --time=6-12:00
#SBATCH --job-name c2b_retrieval
#SBATCH --output=/home/ldang05/scratch/sbatch_outputs/out_sbatch_%j.txt
#SBATCH --mail-type=FAIL

source ~/venv/ss-env/bin/activate
cd /home/ldang05/Starships_prj/Starships_notebooks/Retrieval_examples/

# !!!!!!! Don't forget to remove #SBATCH --array=1-10  !!!!!!!!!
echo "Starting Retrieval code..."
python retrieval_CoRoT_2b.py

## Burnin (use with >>> #SBATCH --array=1-10 for example)
#echo "Starting python code with SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
#python retrieval_WASP-33b_JR_guillot_spirou_only.py idx_file=$SLURM_ARRAY_TASK_ID
