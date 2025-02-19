#!/bin/bash
#SBATCH --job-name=ds_inference
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

export CUDA_VISIBLE_DEVICES=""

# Load required modules
module use /sw/modulefiles
module load openmpi

# (Optional) Activate Conda environment
# conda activate /scratch/user03/miniconda/envs/deepspeed_cpu

# Set up the rendezvous address: pick the first node’s hostname.
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR
export MASTER_PORT=29500  # ensure this port is free

#while true; do
#    echo "Resource usage at $(date):" >> resource_usage.log
#    echo "Top CPU processes:" >> resource_usage.log
#    ps aux --sort=-%cpu | head -n 10 >> resource_usage.log
#    echo "Memory usage:" >> resource_usage.log
#    free -h >> resource_usage.log
#    echo "-------------------------------------" >> resource_usage.log
#    sleep 60
#done &
#MONITOR_PID=$!

srun --distribution=cyclic --cpu-bind=cores -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} \
    python -m torch.distributed.run \
      --nnodes=1 \
      --nproc_per_node=2 \
      --rdzv_id=1234 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      /scratch/user03/scripts/llm-inference/multi-node/deepspeed_only/containarized/new_multi_node_debug.py \
      --deepspeed /scratch/user03/scripts/llm-inference/multi-node/deepspeed_only/containarized/new_ds_config.json

#kill $MONITOR_PID
