# Running PeakNet on Frontier Interactive Nodes

This guide covers how to run the PeakNet training code on Frontier's interactive nodes, both in single-rank and multi-rank configurations.

## Prerequisites

### 1. Request an Interactive Node
```bash
# Request an interactive node with 1 GPU for 2 hours
salloc -A <your_project_id> -t 02:00:00 -p batch -N 1 --ntasks-per-node=1 --gpus=1

# Or request all 8 GPUs on a node
salloc -A <your_project_id> -t 02:00:00 -p batch -N 1 --ntasks-per-node=8 --gpus=8
```

### 2. Load Required Modules
Once on the interactive node:
```bash
module load PrgEnv-amd
module load rocm/6.2.4
module load cray-mpich
```

### 3. Set Environment Variables
```bash
# Enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software

# PyTorch/ROCm optimizations
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1

# Set visible devices (adjust based on your allocation)
export ROCR_VISIBLE_DEVICES=0  # For single GPU
# export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # For all 8 GPUs

# Optional: Enable debug output
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO
```

## Single-Rank Execution

### Direct Python Execution
The simplest way to run on a single GPU:

```bash
# Using Hydra-based launcher
python train_hiera_seg.py \
    job=my_test_job \
    train_config=frontier_test \
    +trainer.num_gpus=1 \
    +trainer.batch_size=32

# Using legacy YAML config
python train_hiera_seg.py experiments/yaml/my_config.yaml
```

### Using srun (Single Rank)
For consistency with multi-node jobs:

```bash
srun --ntasks=1 \
     --cpus-per-task=7 \
     --gpus-per-task=1 \
     --gpu-bind=closest \
     python train_hiera_seg.py \
         job=my_test_job \
         train_config=frontier_test
```

## Multi-Rank Execution (Single Node)

### Running on All 8 GPUs
To use all 8 GPUs on the interactive node:

```bash
# Set distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Launch with srun
srun --ntasks=8 \
     --ntasks-per-node=8 \
     --cpus-per-task=7 \
     --gpus-per-task=1 \
     --gpu-bind=closest \
     python train_hiera_seg.py \
         job=my_test_job \
         train_config=frontier_test \
         +trainer.num_gpus=8 \
         +trainer.batch_size=256  # Scale batch size with GPUs
```

### Running on Subset of GPUs (e.g., 4 GPUs)
```bash
# Limit visible devices
export ROCR_VISIBLE_DEVICES=0,1,2,3

# Launch with 4 tasks
srun --ntasks=4 \
     --cpus-per-task=7 \
     --gpus-per-task=1 \
     --gpu-bind=closest \
     python train_hiera_seg.py \
         job=my_test_job \
         train_config=frontier_test \
         +trainer.num_gpus=4
```

## Using MPI Directly

For more control, you can use mpirun:

```bash
# Single rank
mpirun -n 1 python train_hiera_seg.py train_config=frontier_test

# Multi-rank (8 GPUs)
mpirun -n 8 \
    --bind-to core \
    --map-by ppr:8:node \
    python train_hiera_seg.py train_config=frontier_test
```

## Quick Test Scripts

### 1. Test GPU Detection
```bash
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### 2. Test Single GPU Training
```bash
python test_single_gpu.py
```

### 3. Test Multi-GPU with PyTorch DDP
Create a simple test script:
```python
# test_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # Initialize distributed training
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        rank = 0
        world_size = 1
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set device
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    print(f"Rank {rank}/{world_size} using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
    
    # Create model
    model = torch.nn.Linear(10, 10).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Test forward pass
    x = torch.randn(32, 10).to(device)
    y = model(x)
    print(f"Rank {rank}: Output shape {y.shape}")
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run it:
```bash
# Single GPU
python test_ddp.py

# Multi-GPU
srun -n 8 python test_ddp.py
```

## Troubleshooting

### 1. GPU Not Detected
```bash
# Check if ROCm is properly loaded
rocm-smi

# Check PyTorch ROCm support
python -c "import torch; print(torch.version.hip)"
```

### 2. MPI Errors
```bash
# Ensure MPI is properly initialized
which mpirun
ldd $(which python) | grep mpi

# Test MPI
mpirun -n 2 hostname
```

### 3. Memory Issues
```bash
# Monitor GPU memory
watch -n 1 rocm-smi

# In Python, check memory
python -c "import torch; print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
```

### 4. Performance Debugging
```bash
# Enable profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ROCm profiling
export AMD_LOG_LEVEL=4
```

## Best Practices

1. **Always specify ROCR_VISIBLE_DEVICES** to control which GPUs are used
2. **Use --gpu-bind=closest** for optimal GPU-CPU affinity
3. **Set OMP_NUM_THREADS=1** to avoid CPU oversubscription
4. **Scale batch size with number of GPUs** for optimal throughput
5. **Use NCCL for multi-GPU communication** (PyTorch default)

## Example: Full Training Command

```bash
# Single GPU training
python train_hiera_seg.py \
    job=frontier_test_single \
    train_config=frontier_test \
    +checkpoint.save_dir=$MEMBERWORK/<project>/checkpoints \
    +dataset.data_dir=$PROJWORK/<project>/data \
    +trainer.num_gpus=1

# 8 GPU training
srun -n 8 --gpus-per-task=1 --gpu-bind=closest \
    python train_hiera_seg.py \
    job=frontier_test_multi \
    train_config=frontier_test \
    +checkpoint.save_dir=$MEMBERWORK/<project>/checkpoints \
    +dataset.data_dir=$PROJWORK/<project>/data \
    +trainer.num_gpus=8 \
    +trainer.batch_size=256
```

## Monitoring Training

```bash
# Watch GPU utilization
watch -n 1 rocm-smi

# Monitor training logs
tail -f slurm-*.out

# Check checkpoint directory
ls -la $MEMBERWORK/<project>/checkpoints/
```