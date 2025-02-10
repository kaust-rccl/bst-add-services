#!/bin/bash -l

#SBATCH --job-name=bench-tinyIM-ds
#SBATCH --partition=a100n       # The partition, as seen in output of sinfo. This chooses the GPU pool
#SBATCH --gpus=8		# Total number of GPUs
#SBATCH --gpus-per-node=8	# Number of GPUs per compute node (machine)
#SBATCH --ntasks=8		# Number of CPU processes lanuching work on each GPU (should be equal to the total number of GPUs)
#SBATCH --tasks-per-node=8      # Number of CPU processes on each compute node (machine) (should be equal to the number of GPUs per node)
#SBATCH --cpus-per-task=12      # Number of worker processes for Dataloading feeding to each GPU.
#SBATCH --hint=nomultithread    # Disabling hyperthreading
#SBATCH --time=00:30:00	        # Request for time for the job to run and then terminate automatically (HH:MM:SS)


scontrol show job ${SLURM_JOBID}
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
	--launcher SLURM \
	--launcher_args "\--mpi=pmix" \
	--num_nodes=${SLURM_NNODES} \
	--num_gpus=${SLURM_GPUS_PER_NODE} \
       	../scripts/train_resnet50_ds.py \
	--num-workers=${SLURM_CPUS_PER_TASK} \
	--deepspeed_config ./ds_config.json \
	--epochs=1 --log-interval 100 
end=$(date +%s)
echo "Elapsed Time for ${SLURM_GPUS} GPUs run on partition ${SLURM_JOB_PARTITION} : $(($end-$start)) seconds"
