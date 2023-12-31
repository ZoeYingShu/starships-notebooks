#!/bin/bash
#SBATCH --account=def-dlafre
#SBATCH --nodes=5
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=1-12:00
#SBATCH --job-name logl_map_c2b
#SBATCH --output=/home/ldang05/scratch/sbatch_outputs/out_sbatch_%j.txt
#SBATCH --mail-type=FAIL

source ~/venv/ss-env/bin/activate
cd /home/ldang05/Starships_prj/Starships_notebooks/Retrieval_examples/

echo "Starting logl_map code..."
python logl_map_detailed_corot_2b.py
