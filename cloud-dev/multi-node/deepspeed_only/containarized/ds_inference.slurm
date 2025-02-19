#!/bin/bash
#SBATCH --job-name=ds_inference
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

export CUDA_VISIBLE_DEVICES=""


# Load required modules
module use /sw/modulefiles
module load openmpi

# Activate Conda environment

#conda activate /scratch/user03/miniconda/envs/deepspeed_cpu

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500  # Ensure port is free

srun --distribution=cyclic --cpu-bind=cores -n 2 -N 2 -c ${SLURM_CPUS_PER_TASK} \
     python -m torch.distributed.run \
     --nnodes=2 \
     --nproc_per_node=1 \
     --rdzv_id=1234 \
     --rdzv_backend=c10d \
     --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
     deepspeed_multi_nodes.py \
     --deepspeed /scratch/user03/scripts/llm-inference/multi-node/deepspeed_only/containarized/ds_config.json
