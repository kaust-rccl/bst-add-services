#!/bin/bash -l

#SBATCH --job-name=bench-tinyIM-ds
#SBATCH --partition=a100n
#SBATCH --gpus=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00


scontrol show job ${SLURM_JOBID}
#rm -rf /home/shaima0d/.cache/torch_extensions/*
module use /sw/modulefiles
module load python
module load openmpi
export TRITON_CACHE_DIR=${PWD}/cache
export EPOCHS=1
export DATA_DIR="/workspace/datasets/tiny-imagenet-200"
export OMP_NUM_THREADS=1
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

export HOSTFILE=./hostfile
rm $HOSTFILE
for i in ${nodes_array[*]}
do
	echo $i slots=${SLURM_GPUS_PER_NODE} >> ${HOSTFILE}
done

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
export master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -i)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${MASTER_PORT}"


workers=${SLURM_CPUS_PER_TASK}

echo "Hostname: $(/bin/hostname)"
echo "CPU workers: $workers"

start=$(date +%s)
deepspeed --hostfile ./hostfile \
	--launcher OPENMPI \
	--num_nodes=${SLURM_NNODES} \
       	../scripts/train_resnet50_ds.py --num-workers=${SLURM_CPUS_PER_TASK} \
	--deepspeed_config ./ds_config.json \
	--epochs=1 --log-interval 100 
end=$(date +%s)
echo "Elapsed Time for ${SLURM_GPUS} GPUs run on partition ${SLURM_JOB_PARTITION} : $(($end-$start)) seconds"
	#--num_gpus=${SLURM_GPUS_PER_NODE} \
