#!/bin/bash
#SBATCH -n 1
#SBATCH --partition=t4
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00

srun echo "Howdy, I am $(/bin/hostname)"
lscpu
srun nvidia-smi
srun nvidia-smi topo -m

