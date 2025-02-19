#!/bin/bash
#SBATCH --job-name=test_rendezvous
#SBATCH --output=test_rendezvous-%j.out
#SBATCH --error=test_rendezvous-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

export CUDA_VISIBLE_DEVICES=""

# Set the rendezvous (master) address and port.
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500

srun --distribution=cyclic --cpu-bind=cores -n 2 -N 2 python -m torch.distributed.run \
    --nnodes=2 \
    --nproc_per_node=1 \
    --rdzv_id=1234 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    test_rendez_vous.py

