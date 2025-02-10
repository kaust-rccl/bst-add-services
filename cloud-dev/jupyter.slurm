#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=dev
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00

module use /sw/modulefiles
module load python

USER=$1
export SCRATCH=/workspace/$USER

export JUPYTER_CONFIG_DIR=${SCRATCH}/.jupyter
export JUPYTER_DATA_DIR=${SCRATCH}/.local/share/jupyter
export JUPYTER_RUNTIME_DIR=${SCRATCH}/.local/share/jupyter/runtime
export IPYTHONDIR=${SCRATCH}/.ipython
export XDG_CACHE_HOME=${SCRATCH}/.cache
export NLTK_DATA=${SCRATCH}

echo $USER scratch $SCRATCH
node=$(hostname -s)
submit_host=${SLURM_SUBMIT_HOST}
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo ${node} pinned to port ${port} on ${SLURM_SUBMIT_HOST}

echo -e "
Copy the following line in a new terminal to create a SSH tunnel:
ssh -L ${port}:${node}:${port} <username>@<ip-address-headnode>

Copy the URL provided below by jupyter-server (one starting with http://127.0.0.1/...) and paste it in your browser on your workstation/laptop.

Do not forget to close the notebooks you open in you browser and shutdown the jupyter client in your browser for gracefully exiting this job or else you will have to manually cancel this job running your jupyter server.
"

jupyter lab --no-browser --port=${port} --port-retries=0  --ip=${node} --NotebookApp.token=''
