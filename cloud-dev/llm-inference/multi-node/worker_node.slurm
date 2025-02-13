#!/bin/bash
#SBATCH --job-name=ray_worker
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

conda activate /scratch/user03/miniconda/envs/venv

export NUM_CPUS_PER_NODE=${SLURM_CPUS_PER_TASK}

export ip_head=$(cat ./head_node_info | cut -d " " -f 1)
export head_node_ip=$(echo ${ip_head} | cut -d ":" -f 1)
export redis_password=$(cat ./head_node_info | cut -d " " -f 2)
export dashboard_port=$(cat ./head_node_info | cut -d " " -f 3)

ray start --address ${ip_head} \
	--num-cpus ${NUM_CPUS_PER_NODE}\
	--block &
sleep 20

ray status --address ${ip_head}
sleep 10

if [ -f "shutdown.txt" ] ; then
  echo " Stopping ray on Node: $(/bin/hostname)"
  ray stop
else
  while [ ! -f "shutdown.txt" ]; 
   do
     sleep 20
   done   
fi


