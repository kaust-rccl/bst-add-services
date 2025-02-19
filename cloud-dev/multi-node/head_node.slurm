#!/bin/bash
#SBATCH --job-name=ray_head
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Load modules (adjust as needed)
module use /sw/modulefiles
module load python
module load openmpi

# Set free ports for Ray and dashboard
export server_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export dashboard_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Get head node IP and set environment variables
head_node_ip=$(hostname -i)
export ip_head="${head_node_ip}:${server_port}"

# Write head node connection info to a shared location (ensure /scratch is accessible by all nodes)
echo "${ip_head}  ${dashboard_port}" > /scratch/user03/scripts/llm-inference/multi-node/head_node_info

# Start Ray head (with no compute allocation here)
ray start --node-ip-address ${head_node_ip} --port ${server_port} --head \
          --dashboard-port ${dashboard_port} --dashboard-host=127.0.0.1 \
          --num-cpus 0 --block &
sleep 20

# Optionally, check that the head is up and then submit the inference job.
echo "Ray head started on ${head_node_ip}:${server_port}. Submitting inference job..."
echo "exclude nodes: ${SLURM_NODELIST}"
sbatch -x $SLURM_NODELIST worker_node.slurm

# Optionally, wait or keep the head running as needed.
wait

