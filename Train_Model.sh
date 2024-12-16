#!/bin/bash -l        
#SBATCH --time=3:00:00
#SBATCH --ntasks=4       
#SBATCH --cpus-per-task=1  
#SBATCH --mem=50G
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1

cd /users/9/lee02328/Ada_Comp/SST_diffusion

# Load Miniconda
source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

conda activate AVIT2

python Train_model.py