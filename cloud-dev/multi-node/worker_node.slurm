#!/bin/bash
#SBATCH --job-name=inference_launch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00


# Load required modules (adjust as needed)
module use /sw/modulefiles
module load python
module load openmpi

# Activate the conda environment (adjust the path to your environment)
#source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/user03/miniconda/envs/myenv

# Wait until the head node info is available
while [ ! -f /scratch/user03/scripts/llm-inference/multi-node/head_node_info ]; do
    echo "Waiting for head_node_info to be available..."
    sleep 10
done

# Read head node info
head_info=$(cat /scratch/user03/scripts/llm-inference/multi-node/head_node_info)
head_ip=$(echo ${head_info} | cut -d " " -f 1 | cut -d ":" -f 1)
server_port=$(echo ${head_info} | cut -d " " -f 1 | cut -d ":" -f 2)
# Optionally, read redis_password and dashboard_port if needed

# Start Ray on each worker node so it joins the cluster.
# This command registers the worker with the head node.
ray start --address ${head_ip}:${server_port} --num-cpus ${SLURM_CPUS_PER_TASK} --block &
sleep 10

# Launch distributed inference across 2 nodes.
# Using srun to launch 2 tasks (one per node).
# Each process is given a unique node rank (using SLURM_NODEID) to coordinate the torch distributed group.
srun --distribution=cyclic --cpu-bind=cores -n 2 -N 2 -c ${SLURM_CPUS_PER_TASK} \
     python -m torch.distributed.launch --use_env \
     --nproc_per_node=1 \
     --nnodes=2 \
     --node_rank=$SLURM_NODEID \
     --master_addr=${head_ip} \
     --master_port=${server_port} \
     multinode_inference.py

