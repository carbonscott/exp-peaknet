#!/bin/bash
#SBATCH --output=slurm/{{ job }}.%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/{{ job }}.%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account={{ account }}             # Your ORNL project ID
#SBATCH --partition={{ partition }}         # batch is the main partition
#SBATCH --time={{ walltime }}               # Walltime limit
#SBATCH --job-name={{ job }}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --ntasks-per-node={{ num_gpus_per_node }}  # 8 for Frontier (one task per GCD)
#SBATCH --cpus-per-task={{ num_cpus_per_task }}    # 7 due to low-noise mode
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
{%- if exclude_nodes %}
#SBATCH --exclude={{ exclude_nodes }}
{%- endif %}
{%- if qos %}
#SBATCH --qos={{ qos }}
{%- endif %}
{%- if core_spec_override %}
#SBATCH -S {{ core_spec_override }}  # Override default core specialization
{%- endif %}
{%- if mail_type %}
#SBATCH --mail-type={{ mail_type }}
{%- endif %}
{%- if mail_user %}
#SBATCH --mail-user={{ mail_user }}
{%- endif %}

# Load required modules for Frontier
module load PrgEnv-amd
module load rocm/{{ rocm_version }}
module load cray-mpich

# Set up environment for AMD GPUs - let SLURM handle GPU assignment
# export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # All 8 GCDs per node
export MPICH_GPU_SUPPORT_ENABLED=1            # Enable GPU-aware MPI
export FI_MR_CACHE_MONITOR=memhooks            # Recommended for GPU-aware MPI
export FI_CXI_RX_MATCH_MODE=software           # Recommended for GPU-aware MPI

# PyTorch/ROCm environment variables
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export NCCL_DEBUG={{ nccl_debug }}
export TORCH_NCCL_BLOCKING_WAIT=0

# Fix "Address family not supported by protocol" errno 97 errors (critical for Frontier)
export NCCL_SOCKET_FAMILY=AF_INET  # Force IPv4-only communication
export NCCL_IB_DISABLE=1           # Disable InfiniBand if causing issues

# Set up Huggingface cache
export TRANSFORMERS_CACHE={{ transformers_cache }}

# Configure MIOpen cache to avoid SQLite database disk I/O errors (ORNL Frontier fix)
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-$USER-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# OpenMP settings
export OMP_NUM_THREADS={{ OMP_NUM_THREADS }}

# Set up preemption checkpoint handling
export PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
export PREEMPT_METADATA_PATH="$PREEMPT_ROOT/{{ job }}"

# Check if using hydra_command (new style) or yaml_config (old style)
{%- if hydra_command %}
# Using Hydra command directly
LAUNCH_CMD="{{ hydra_command }}"

# Add checkpoint resume if exists
if [ -f $PREEMPT_METADATA_PATH ]; then
    echo "Resuming from preemptive checkpoint..."
    CHECKPOINT_PATH=$(cat $PREEMPT_METADATA_PATH)
    LAUNCH_CMD="${LAUNCH_CMD} checkpoint.path_chkpt_prev=${CHECKPOINT_PATH}"
fi
{%- else %}
# Using YAML config file (legacy mode)
LAUNCH_CMD="python {{ trainer }} {{ yaml_config }}"

# Handle checkpoint resume for YAML mode
if [ -f $PREEMPT_METADATA_PATH ]; then
    echo "Resuming from preemptive checkpoint..."
    python -c "import yaml, os; data = yaml.safe_load(open('{{ yaml_config }}')); data['checkpoint']['path_chkpt_prev'] = open(os.getenv('PREEMPT_METADATA_PATH')).read().strip(); yaml.safe_dump(data, open('{{ yaml_config }}', 'w'))"
fi
{%- endif %}

# Get head node information for distributed training
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | head -n1)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=23456

# Log environment
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "Launch command: $LAUNCH_CMD"
export LOGLEVEL=INFO

# Launch distributed training with srun
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node={{ num_gpus_per_node }} \
     --cpus-per-task={{ num_cpus_per_task }} \
     --gpus-per-task=1 \
     --gpu-bind=closest \
     bash -c "$LAUNCH_CMD"