#!/bin/bash
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls:prjdat21         # Check it in your Iris portal: https://iris.nersc.gov
#SBATCH --partition={{partition}}
#SBATCH --time {{ walltime }}          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --exclusive
#SBATCH --job-name={{ job }}
#SBATCH --gres=gpu:{{ num_gpus_per_node }}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --ntasks={{ num_tasks }}
#SBATCH --cpus-per-task={{ num_cpus_per_task }}

# Set up the Huggingface's cache directory
export TRANSFORMERS_CACHE={{ transformers_cache }}

export OMP_NUM_THREADS={{ OMP_NUM_THREADS }}
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

# Set up a meta checkpoint file
export PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
export PREEMPT_METADATA_PATH="$PREEMPT_ROOT/{{ job }}"

# Check if a checkpoint exists and resume from it
if [ -f $PREEMPT_METADATA_PATH ]; then
    echo "Resuming from preemptive checkpoint..."
    python -c "import yaml, os; data = yaml.safe_load(open('experiments/yaml/{{ job }}.yaml')); data['checkpoint']['path_chkpt_prev'] = open(os.getenv('PREEMPT_METADATA_PATH')).read().strip(); yaml.safe_dump(data, open('experiments/yaml/{{ job }}.yaml', 'w'))"
fi

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

mpirun -n {{ num_tasks }} python {{ trainer }} {{ yaml_config }}
