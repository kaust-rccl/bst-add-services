#!/bin/bash
#SBATCH --job-name=llm_inference
#SBATCH --output=llm_inference.out
#SBATCH --error=llm_inference.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

conda activate /scratch/user03/miniconda/envs/venv

export CUDA_VISIBLE_DEVICES=-1

python3 /scratch/user03/llm-inference/llm_inference.py
