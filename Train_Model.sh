#!/bin/bash -l        
#SBATCH --time=6:00:00
#SBATCH --ntasks=4       
#SBATCH --cpus-per-task=1  
#SBATCH --mem=10G
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1

cd /scratch.global/lee02328/noaa_sst_data/

# Load Miniconda
source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

rm logs/*

conda activate AVIT2

python Train_model.py