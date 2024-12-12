#!/bin/bash -l        
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=lee02328@umn.edu


cd /scratch.global/lee02328/noaa_sst_data

# Load Miniconda
source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

conda activate /home/boleydl/lee02328/miniconda3/envs/GenAI

python Data_Prep.py