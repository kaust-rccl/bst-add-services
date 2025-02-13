#!/bin/bash
#SBATCH --job-name=ray_head
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

# Activate Conda environment
source /scratch/user03/miniconda/bin/activate /scratch/user03/miniconda/envs/venv


if [ -z ${NUM_WORKERS} ] ; then
  NUM_WORKERS=1
else
  NUM_WORKERS=${NUM_WORKERS}
fi

export server_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export dashboard_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

export node=$(/bin/hostname -i)


export redis_password=${SLURM_JOBID}
export head_node_ip=$(hostname -i)
export ip_head=${head_node_ip}:${server_port}
echo "${ip_head} ${redis_password} ${dashboard_port}" > head_node_info

ray start --node-ip-address ${head_node_ip} --port ${server_port} --head  \
	--dashboard-port ${dashboard_port} --dashboard-host=127.0.0.1 \
        --num-cpus 1 --block &

job_ids=()
for (( i=1; i<=${NUM_WORKERS}; i++ ))
do
	job_ids[$i]=$(sbatch -x $SLURM_NODELIST worker_node.slurm | cut -d " " -f 4)
done 
 

while [ ! -z $(squeue -n ray_worker -t PD -h -o %A) ]
do
	echo "Waiting for worker(s) to start"
        sleep 20
done

python -u llm_inference.py

touch $PWD/shutdown.txt
sleep 20
echo " Stopping ray on Head node: $(/bin/hostname)"
ray stop
rm $PWD/shutdown.txt

