#!/bin/bash
#SBATCH --job-name=multi_node_inference
#SBATCH --output=multi_node_inference-%j.out
#SBATCH --error=multi_node_inference-%j.err
#SBATCH --nodes=3               # All nodes allocated in one job
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Load required modules
module use /sw/modulefiles
module load python
module load openmpi


export server_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export dashboard_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export torch_master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Get head node IP (all nodes see the same head info because they are in one allocation)
head_node_ip=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export head_node_ip

# Only the head node starts Ray head
if [ "$SLURM_NODEID" -eq 0 ]; then
  echo "${head_node_ip}:${server_port}" > head_node_info  # shared file for workers
  echo "Starting Ray head on ${head_node_ip}:${server_port}"
 
  ray start --head --port ${server_port} --dashboard-port ${dashboard_port} \
            --node-ip-address ${head_node_ip} --num-cpus 0 --block &
  sleep 20
fi

# Ensure the head node info is available (workers may wait)
while [ ! -f head_node_info ]; do sleep 1; done

# Each node now launches its torch distributed process via srun.
# We use SLURM_NODEID and SLURM_JOB_NUM_NODES to set rank and world size.
srun --export=ALL --distribution=cyclic --cpu-bind=cores -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} \
    python -m torch.distributed.launch --use_env \
      --nproc_per_node=1 \
      --nnodes=${SLURM_JOB_NUM_NODES} \
      --node_rank=${SLURM_NODEID} \
      --master_addr=${head_node_ip} \
      --master_port=${torch_master_port} \
      my_inference_script.py --deepspeed "$@" &> $PWD/run_${SLURM_JOBID}_${SLURM_NODEID}.txt &


wait

# After inference completes, stop Ray.
ray stop

