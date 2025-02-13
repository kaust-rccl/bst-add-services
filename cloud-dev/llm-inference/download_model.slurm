#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --output=download_model.out
#SBATCH --error=download_model.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

conda activate /scratch/user03/miniconda/envs/venv

python3 /scratch/user03/llm-inference/download_model.py

