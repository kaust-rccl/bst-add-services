#!/bin/bash
#SBATCH --job-name=ray_worker
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu

module load /sw/modulefiles
module load python

export NUM_CPUS_PER_NODE=${SLURM_CPUS_PER_TASK}
export NUM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE}

export ip_head=$(cat ./head_node_info | cut -d " " -f 1)
export head_node_ip=$(echo ${ip_head} | cut -d ":" -f 1)
export redis_password=$(cat ./head_node_info | cut -d " " -f 2)
export dashboard_port=$(cat ./head_node_info | cut -d " " -f 3)


ray start --address ${ip_head}  --redis-password ${redis_password} \
	--num-cpus ${NUM_CPUS_PER_NODE} --num-gpus ${NUM_GPUS_PER_NODE} \
	--block &
sleep 20
ray status --address ${ip_head} --redis_password ${redis_password}
sleep 10

# worker shutdown strategy
if [ -f "shutdown.txt" ] ; then
  echo " Stopping ray on Node: $(/bin/hostname)"
  ray stop
else
  while [ ! -f "shutdown.txt" ]; 
   do
     sleep 20
   done   
fi
