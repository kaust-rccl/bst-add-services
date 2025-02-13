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
export node=$(/bin/hostname -i)


# Get head node IP (all nodes see the same head info because they are in one allocation)
head_node_ip=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export head_node_ip

# Only the head node starts Ray head
if [ "$SLURM_NODEID" -eq 0 ]; then
  echo "${head_node_ip}:${server_port}" > head_node_info  # shared file for workers
  echo "Starting Ray head on ${head_node_ip}:${server_port}"
  
  ray start --head --port ${server_port} --dashboard-port ${dashboard_port} \
            --node-ip-address ${head_node_ip} --dashboard-host=127.0.0.1 --num-cpus 0 --block &
  sleep 20
fi


