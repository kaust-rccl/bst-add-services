#!/bin/bash
#SBATCH --job-name=dummy_job
#SBATCH --output=output_%x-%j.out
#SBATCH --error=output_%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=2
#SBATCH --time=00:30:00


# Load required modules
module use /sw/modulefiles
module load python
module load openmpi


export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345

srun --distribution=cyclic --cpu-bind=cores -n 1 -N 1 \
        python -m torch.distributed.run --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_NTASKS_PER_NODE \
        --rdzv_id=1234 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        python_dummy.py  > my_output.log 2>&1

