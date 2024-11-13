#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=lcls:prjdat21
#SBATCH --nodes=2
#SBATCH --gpus-per-node=10
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=slurm/inference-s3df-issue10-atto-2bifpn512-1.0.%j.log
#SBATCH --error=slurm/inference-s3df-issue10-atto-2bifpn512-1.0.%j.err

conda.ml
cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet

# Export necessary environment variables
export SLURM_NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
export HEAD_NODE=$(hostname)
export WORKER_NODES=$(scontrol show hostname $SLURM_JOB_NODELIST | grep -v $HEAD_NODE)

# Write node information to a file that can be used by the launcher
echo "HEAD_NODE=$HEAD_NODE" > node_info.txt
echo "WORKER_NODES=$WORKER_NODES" >> node_info.txt

set -x

# Launch the job coordinator script
bash run_inference.nersc-issue10-atto-2bifpn512-1.0.launch.sh
