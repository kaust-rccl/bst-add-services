#!/bin/bash
#SBATCH --job-name=ds_inference
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

export CUDA_VISIBLE_DEVICES=""


# Load required modules
module use /sw/modulefiles
module load openmpi

# Activate Conda environment

#conda activate /scratch/user03/miniconda/envs/deepspeed_cpu

python3 /scratch/user03/scripts/llm-inference/multi-node/deepspeed_only/containarized/deepspeed_single_node.py 
