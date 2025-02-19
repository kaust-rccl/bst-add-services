#!/bin/bash
#SBATCH --job-name=ds_inference
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:30:00

export CUDA_VISIBLE_DEVICES=""


# Load required modules
module use /sw/modulefiles
module load python
module load openmpi

# Activate Conda environment
#conda activate /scratch/user03/miniconda/envs/myenv

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500  # Ensure port is free
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun --distribution=cyclic --cpu-bind=cores -n 2 -N 2 \
     python -m torch.distributed.run \
     --nnodes=2 \
     --nproc_per_node=1 \
     deepspeed_multi_nodes.py \
     --deepspeed ds_config.json
