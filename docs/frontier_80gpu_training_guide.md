# Frontier 80-GPU Training Guide (10 Nodes)

This guide covers how to run both ConvNeXt and Hiera models on 10 nodes with 80 GPUs using the Hydra configuration system.

## Quick Reference

### ConvNeXt V2 Huge (659M parameters)
```bash
python launch_unified_hydra.py \
    job=convnext_huge_production \
    train_config=frontier_convnext_huge_1000step_80gpus \
    resource_configs=frontier_10node \
    trainer=train_convnext_seg.py \
    qos=normal \
    walltime="04:00:00" \
    auto_submit=true
```

### Hiera Huge (production)
```bash
python launch_unified_hydra.py \
    job=hiera_huge_production \
    train_config=frontier_hiera_huge_1000step_80gpus \
    resource_configs=frontier_10node \
    trainer=train_hiera_seg.py \
    qos=normal \
    walltime="04:00:00" \
    auto_submit=true
```

## Detailed Setup Instructions

### 1. Available Configurations

#### ConvNeXt Configurations:
- **Base:** `train_config=convnext` (smaller model for testing)
- **Huge:** `train_config=frontier_convnext_huge_1000step_80gpus` (659M parameters)

#### Hiera Configurations:
- **Base:** `train_config=hiera` (smaller model for testing)
- **Huge:** `train_config=frontier_hiera_huge_1000step_80gpus` (production huge model)

#### Resource Configuration:
- **80 GPUs:** `resource_configs=frontier_10node` (10 nodes × 8 GPUs each)

### 2. Step-by-Step Process

#### Step 1: Generate Configuration
```bash
# For ConvNeXt Huge
python launch_unified_hydra.py \
    job=my_convnext_run \
    train_config=frontier_convnext_huge_1000step_80gpus \
    resource_configs=frontier_10node \
    trainer=train_convnext_seg.py \
    auto_submit=false  # Generate files but don't submit yet
```

#### Step 2: Review Generated Files
```bash
# Check the generated YAML config
cat experiments/yaml/my_convnext_run.yaml

# Check the generated SLURM script
cat experiments/jobs/my_convnext_run.frontier.sbatch
```

#### Step 3: Submit Job
```bash
# Method 1: Auto-submit during generation
python launch_unified_hydra.py \
    job=my_convnext_run \
    train_config=frontier_convnext_huge_1000step_80gpus \
    resource_configs=frontier_10node \
    trainer=train_convnext_seg.py \
    auto_submit=true

# Method 2: Manual submission
sbatch experiments/jobs/my_convnext_run.frontier.sbatch
```

### 3. Common Overrides

#### Production Settings
```bash
python launch_unified_hydra.py \
    job=convnext_production \
    train_config=frontier_convnext_huge_1000step_80gpus \
    resource_configs=frontier_10node \
    trainer=train_convnext_seg.py \
    qos=normal \                          # Use normal queue instead of debug
    walltime="08:00:00" \                 # 8 hours for longer training
    lr_scheduler.total_steps=10000 \      # More training steps
    checkpoint.chkpt_saving_steps=100 \   # Save checkpoints every 100 steps
    auto_submit=true
```

#### Development/Testing Settings
```bash
python launch_unified_hydra.py \
    job=convnext_test \
    train_config=frontier_convnext_huge_1000step_80gpus \
    resource_configs=frontier_10node \
    trainer=train_convnext_seg.py \
    qos=debug \                           # Debug queue (faster scheduling)
    walltime="02:00:00" \                 # 2 hours for testing
    lr_scheduler.total_steps=100 \        # Short test run
    dataset.batch_size=8 \                # Smaller batch size for testing
    auto_submit=false                     # Review before submitting
```

### 4. Model Architecture Specifications

#### ConvNeXt V2 Huge
- **Parameters:** 659M
- **Architecture:** `[352, 704, 1408, 2816]` hidden sizes, `[3, 3, 27, 3]` depths
- **Memory:** Higher memory usage, FSDP zero3 recommended
- **Batch Size:** 16 per node (with gradient accumulation)
- **Input Size:** 384×384

#### Hiera Huge  
- **Parameters:** ~580M (estimated)
- **Architecture:** `[2, 6, 48, 4]` stages, `192` embed_dim
- **Memory:** More memory efficient than ConvNeXt
- **Batch Size:** 32 per node
- **Input Size:** 512×512

### 5. Resource Specifications (frontier_10node.yaml)

```yaml
num_nodes: 10                    # 10 Frontier nodes
num_tasks: 80                    # 80 total MPI tasks
num_gpus_per_node: 8            # 8 GCDs per node
num_cpus_per_task: 7            # 7 CPUs per task (low-noise mode)
partition: batch                 # Frontier batch partition
account: "mph121"               # Your account
walltime: "02:00:00"            # Default 2 hours
qos: "debug"                    # Default debug queue
```

### 6. Monitoring and Debugging

#### Check Job Status
```bash
squeue -u $USER                 # Check job queue status
squeue -j <job_id> --format="%.18i %.9P %.12j %.8u %.8T %.10M %.6D %R"
```

#### Monitor Training Progress
```bash
# Check latest logs
tail -f experiments/logs/*/rank0.log

# Monitor GPU usage
watch -n 5 'squeue -u $USER -o "%.18i %.12j %.8T %.10M %.6D %C"'
```

#### Debug Common Issues
```bash
# Check SLURM output
cat slurm-<job_id>.out

# Check if all nodes are healthy
sinfo -N -l | grep batch

# Check for failed nodes
squeue -j <job_id> -t PD --format="%.10i %.9P %.12j %.8u %.2t %.10M %.6D %R"
```

### 7. Performance Expectations

#### ConvNeXt V2 Huge (80 GPUs)
- **Training Speed:** ~2-3 steps/minute (estimated)
- **Memory Usage:** ~40-50GB per GPU
- **1000 steps:** ~5-8 hours
- **Throughput:** ~1,280 samples/step (16 batch_size × 80 GPUs)

#### Hiera Huge (80 GPUs)
- **Training Speed:** ~4-5 steps/minute (estimated)  
- **Memory Usage:** ~30-40GB per GPU
- **1000 steps:** ~3-4 hours
- **Throughput:** ~2,560 samples/step (32 batch_size × 80 GPUs)

### 8. Troubleshooting

#### Common Issues and Solutions

**Issue:** Job pending in queue
```bash
# Check reason
squeue -j <job_id> --format="%.10i %.20R"
# Solution: May need to change QOS from debug to normal for longer runs
```

**Issue:** Out of memory errors
```bash
# Reduce batch size
dataset.batch_size=8
# Or increase gradient accumulation
loss.grad_accum_steps=4
```

**Issue:** Communication timeouts
```bash
# Check network connectivity between nodes
# NCCL debugging is already enabled in configs
```

**Issue:** Slow training startup
```bash
# This is normal for huge models - initialization takes 5-10 minutes
# Monitor logs for "Starting training loop" message
```

### 9. Example Commands Summary

#### Quick Test (ConvNeXt)
```bash
python launch_unified_hydra.py job=test_convnext train_config=frontier_convnext_huge_1000step_80gpus resource_configs=frontier_10node trainer=train_convnext_seg.py lr_scheduler.total_steps=10 auto_submit=false
```

#### Quick Test (Hiera)
```bash
python launch_unified_hydra.py job=test_hiera train_config=frontier_hiera_huge_1000step_80gpus resource_configs=frontier_10node trainer=train_hiera_seg.py lr_scheduler.total_steps=10 auto_submit=false
```

#### Production Run (ConvNeXt)
```bash
python launch_unified_hydra.py job=prod_convnext train_config=frontier_convnext_huge_1000step_80gpus resource_configs=frontier_10node trainer=train_convnext_seg.py qos=normal walltime="12:00:00" lr_scheduler.total_steps=50000 auto_submit=true
```

#### Production Run (Hiera)
```bash
python launch_unified_hydra.py job=prod_hiera train_config=frontier_hiera_huge_1000step_80gpus resource_configs=frontier_10node trainer=train_hiera_seg.py qos=normal walltime="12:00:00" lr_scheduler.total_steps=50000 auto_submit=true
```

## Important Notes

1. **Queue Limits:** Debug queue limited to 2 hours, use `qos=normal` for longer runs
2. **Memory:** ConvNeXt Huge uses more memory than Hiera Huge
3. **Initialization:** Huge models take 5-10 minutes to initialize
4. **Checkpointing:** Enabled by default every 10 steps for 1000-step runs
5. **Resumption:** Use `checkpoint.path_chkpt_prev=path/to/checkpoint` to resume
6. **Dataset:** Both configs use `pretrain/train.csv` and `pretrain/eval.csv`

## Files Generated

- `experiments/yaml/<job_name>.yaml` - Final merged configuration
- `experiments/jobs/<job_name>.frontier.sbatch` - SLURM job script  
- `experiments/logs/<timestamp>/` - Training logs
- `experiments/chkpts/<prefix>_<timestamp>_*` - Model checkpoints