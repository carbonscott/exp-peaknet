# Adding a New HPC Facility

This guide shows how to add support for a new HPC facility to the unified launcher.

## Example: Adding ORNL Frontier Support

Suppose we want to add support for ORNL Frontier (a SLURM-based system).

### Step 1: Create Job Template

Create `hydra_config/templates/frontier.sbatch`:

```bash
#!/bin/bash
#SBATCH --output=slurm/{{ job }}.%j.log
#SBATCH --error=slurm/{{ job }}.%j.err  
#SBATCH --account={{ account }}
#SBATCH --partition={{ partition }}
#SBATCH --time {{ walltime }}
#SBATCH --job-name={{ job }}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --ntasks-per-node={{ num_gpus_per_node }}
#SBATCH --gpus-per-node={{ num_gpus_per_node }}
#SBATCH --cpus-per-task={{ num_cpus_per_task }}

# Frontier-specific modules and environment
module load PrgEnv-gnu
module load rocm
module load python

# Set up environment variables
export TRANSFORMERS_CACHE={{ transformers_cache }}
export OMP_NUM_THREADS={{ OMP_NUM_THREADS }}

# Set up checkpoint metadata  
export PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
export PREEMPT_METADATA_PATH="$PREEMPT_ROOT/{{ job }}"

# Check for preemptive checkpoint resumption
if [ -f $PREEMPT_METADATA_PATH ]; then
    echo "Resuming from preemptive checkpoint..."
    python -c "import yaml, os; data = yaml.safe_load(open('{{ yaml_config }}')); data['checkpoint']['path_chkpt_prev'] = open(os.getenv('PREEMPT_METADATA_PATH')).read().strip(); yaml.safe_dump(data, open('{{ yaml_config }}', 'w'))"
fi

# Launch training with Frontier's job launcher
srun python {{ trainer }} {{ yaml_config }}
```

### Step 2: Create Scheduler Configuration

Create `hydra_config/scheduler_configs/frontier.yaml`:

```yaml
# ORNL Frontier SLURM Configuration
account: "PROJECT123"               # Your Frontier project account
partition: "batch"                  # Standard partition
walltime: "24:00:00"               # Default walltime
num_nodes: 1
num_gpus_per_node: 8               # Frontier has 8 GPUs per node
num_cpus_per_task: 7               # Conservative CPU allocation
trainer: 'train_hiera_seg.py'
OMP_NUM_THREADS: 1
transformers_cache: "$HOME/.cache/huggingface"
launcher_cmd: "srun"               # SLURM job launcher
```

### Step 3: Test the New Facility

```bash
# The unified launcher will auto-discover the new templates
python launch_unified.py \
    job=test-frontier \
    train_config=base \
    resource_configs=base \
    auto_submit=false

# Check that frontier script was generated
ls experiments/jobs/test-frontier.frontier.sbatch
```

### Step 4: Submit Job on Frontier

```bash
# When you're on Frontier, submit the Frontier-specific script
sbatch experiments/jobs/test-frontier.frontier.sbatch
```

## That's It! 🎉

**No code changes required.** The unified launcher automatically:
- Discovers the new `frontier.sbatch` template
- Loads the Frontier-specific scheduler config  
- Generates `<job>.frontier.sbatch` scripts
- Includes it in the output options

## General Steps for Any Facility

1. **Create template**: `hydra_config/templates/<facility>.{sbatch|bsub|pbs}`
2. **Create scheduler config**: `hydra_config/scheduler_configs/<facility>.yaml`
3. **Test**: `python launch_unified.py job=test-<facility> ...`
4. **Done!** The facility is now supported

## Template Variables

Common variables available in all templates:

| Variable | Description | Example |
|----------|-------------|---------|
| `{{ job }}` | Job name | `my-experiment` |
| `{{ yaml_config }}` | Path to YAML config | `experiments/yaml/my-experiment.yaml` |
| `{{ trainer }}` | Training script | `train_hiera_seg.py` |
| `{{ walltime }}` | Job walltime | `48:00:00` |
| `{{ num_nodes }}` | Number of nodes | `1` |
| `{{ num_gpus_per_node }}` | GPUs per node | `10` |
| `{{ num_cpus_per_task }}` | CPUs per task | `2` |
| `{{ account }}` | Account/project ID | `lcls:prjdat21` |
| `{{ partition }}` | Queue/partition | `ada` |
| `{{ transformers_cache }}` | HuggingFace cache | `$HOME/.cache/huggingface` |

## Scheduler-Specific Considerations

### SLURM Systems
- Use `#SBATCH` directives
- Job launcher: `srun` or `mpirun`
- File extension: `.sbatch`

### LSF Systems  
- Use `#BSUB` directives
- Job launcher: `jsrun`
- File extension: `.bsub`

### PBS/Torque Systems
- Use `#PBS` directives  
- Job launcher: `mpirun` or custom
- File extension: `.pbs`

## Example Facilities

The project already includes templates for:
- **SLAC S3DF** (`s3df.sbatch`) - SLURM with `mpirun`
- **NERSC Perlmutter** (`nersc.sbatch`) - SLURM with `srun`  
- **ORNL Summit** (`summit.bsub`) - LSF with `jsrun`

Use these as references for creating new facility templates.