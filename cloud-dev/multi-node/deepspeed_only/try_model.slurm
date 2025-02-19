#!/bin/bash
#SBATCH --job-name=try_model
#SBATCH --output=try_model.out
#SBATCH --error=try_model.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

conda activate /scratch/user03/miniconda/envs/myenv

python3 /scratch/user03/scripts/download_model.py

