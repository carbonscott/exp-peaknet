#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -- Basic imports
import os
import yaml
import tqdm
import signal
import argparse
import inspect
import logging
import traceback
import time

from functools  import partial
from contextlib import nullcontext
from datetime   import timedelta
from omegaconf  import OmegaConf

# -- peaknet specific imports
# --- Dataset
from peaknet.datasets.data import (
    PeakNetDatasetConfig,
    PeakNetDataset,
)
from peaknet.tensor_transforms import (
    Pad,
    PolarCenterCrop,
    MergeBatchPatchDims,
    BatchSampler,
    RandomPatch,
    RandomRotate,
    RandomShift,
    InstanceNorm,
)

# --- Model
from peaknet.modeling.hiera_segmentation import HieraSegmentation
from peaknet.modeling.hiera import Hiera, HieraBlock
from peaknet.modeling.hiera_mae import MaskedAutoencoderHiera

# --- Loss
from peaknet.criterion import CategoricalFocalLoss

# --- Others
from peaknet.utils.seed        import set_seed
from peaknet.utils.misc        import is_action_due
from peaknet.utils.checkpoint  import init_checkpointer
from peaknet.utils.dist        import dist_setup
from peaknet.utils.fsdp        import (
    MemoryMaximizer,
    verify_bfloat_support,
    set_sharding_strategy,
    shard_layers,
    backward_prefetch,
    act_chkpt,
)
import logging
import peaknet.utils.logger as logger_utils
from peaknet.utils.data        import wrap_with_torch_dataloader, create_infinite_dataloader
from peaknet.utils.timestamp   import (
    broadcast_timestamp_to_all_ranks,
)
from peaknet.lr_scheduler      import CosineLRScheduler
from peaknet.perf              import Timer
from peaknet.utils.monitor import (
    ActivationMonitor,
    monitor_param_update_metrics,
)
from peaknet.utils.flops import (
    estimate_flops_per_token,
    estimate_mfu_per_iteration,
)
from peaknet.utils.signal import register_handlers
from peaknet.utils.eval import estimate_loss

# -- Imports for monitoring training dynamics
# Note: Removed transformers.activations import as not needed for Hiera

# -- Torch specific imports
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

# -- Configure optimal thread count for distributed training
# This prevents OMP_NUM_THREADS warning and optimizes performance
if 'OMP_NUM_THREADS' not in os.environ:
    # Calculate optimal thread count: total CPU cores / number of processes per node
    # For distributed training, we want to avoid oversubscription
    total_cores = os.cpu_count() or 4
    # Assume 2 processes per node for distributed training (adjust based on your setup)
    processes_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', '2'))
    optimal_threads = max(1, total_cores // processes_per_node)
    torch.set_num_threads(optimal_threads)
    print(f"Set optimal thread count: {optimal_threads} (total_cores={total_cores}, processes_per_node={processes_per_node})")

# -- Distributed Data Parallel (DDP)
from torch.nn.parallel import DistributedDataParallel as DDP

# -- Fully Sharded Data Parallel (FSDP)
# --- Main
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)

# --- Policy wrapper (using modern ModuleWrapPolicy via utilities)
# Note: ModuleWrapPolicy is used in peaknet.utils.fsdp.shard_layers
from packaging import version

# --- Scaler for float16
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# --- Activation checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

# --- Distributed library
import torch.distributed as dist

# -- Debug
# [WARNING] Making it True may throw errors when using float16.
# Invalid gradients are expected to occur during mixed-precision training in
# float16 and anomaly detection will thus report false errors.
# Refer to https://discuss.pytorch.org/t/convolutionbackward0-returned-nan-values-in-its-0th-output/175571/4
torch.autograd.set_detect_anomaly(False)

# -- Reporting specific imports
import colorama
colorama.init(autoreset = True)

# -- Get the logger using familiar pattern
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  COMMAND LINE INTERFACE
# ----------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description = "Load training configuration from a YAML file to a dictionary.")
parser.add_argument("yaml_file", help="Path to the YAML file")
args = parser.parse_args()

# ----------------------------------------------------------------------- #
#  CONFIGURATION
# ----------------------------------------------------------------------- #
# Load CONFIG from YAML using OmegaConf
fl_yaml = args.yaml_file
config = OmegaConf.load(fl_yaml)

# -- Checkpoint
dir_root_chkpt                = config.checkpoint.directory
fl_chkpt_prefix               = config.checkpoint.prefix
path_chkpt_prev               = config.checkpoint.get("path_chkpt_prev", None)
chkpt_saving_steps            = config.checkpoint.chkpt_saving_steps
preempt_metadata_path         = config.checkpoint.get("preempt_metadata_path", os.environ.get('PREEMPT_METADATA_PATH', None))
preempt_chkpt_saving_steps    = config.checkpoint.preempt_chkpt_saving_steps
state_dict_type               = config.checkpoint.state_dict_type
chkpt_offload_to_cpu          = config.checkpoint.offload_to_cpu
chkpt_rank0_only              = config.checkpoint.rank0_only

# -- Dataset
path_dataset_train     = config.dataset.path_train
path_dataset_eval      = config.dataset.path_eval
drop_last_in_sampler   = config.dataset.drop_last_in_sampler
drop_last_in_loader    = config.dataset.drop_last_in_loader
batch_size             = config.dataset.batch_size
seg_size               = config.dataset.seg_size
num_workers            = config.dataset.num_workers
pin_memory             = config.dataset.pin_memory
prefetch_factor        = config.dataset.prefetch_factor
debug_dataloading      = config.dataset.debug
cache_size             = config.dataset.cache_size
# Transform parameters (keep existing)
H_pad = config.dataset.transforms.H_pad
W_pad = config.dataset.transforms.W_pad
Hv = config.dataset.transforms.Hv
Wv = config.dataset.transforms.Wv
sigma = config.dataset.transforms.sigma
num_crop = config.dataset.transforms.num_crop
uses_pad = config.dataset.transforms.set.pad
uses_polar_center_crop = config.dataset.transforms.set.polar_center_crop

# -- Model
from_scratch        = config.model.from_scratch

# Segmentation-specific parameters
num_classes         = config.model.hiera.get("num_classes", 2)
decoder_embed_dim   = config.model.hiera.get("decoder_embed_dim", 512)
decoder_depth       = config.model.hiera.get("decoder_depth", 8)
decoder_num_heads   = config.model.hiera.get("decoder_num_heads", 16)

# Core Hiera architecture parameters
input_size          = tuple(config.model.hiera.get("input_size", [224, 224]))
in_chans            = config.model.hiera.get("in_chans", 1)  # Typically 1 for X-ray data
embed_dim           = config.model.hiera.get("embed_dim", 96)
num_heads           = config.model.hiera.get("num_heads", 1)
stages              = tuple(config.model.hiera.get("stages", [2, 3, 16, 3]))
q_pool              = config.model.hiera.get("q_pool", 3)
q_stride            = tuple(config.model.hiera.get("q_stride", [2, 2]))
mask_unit_size      = tuple(config.model.hiera.get("mask_unit_size", [8, 8]))
mask_unit_attn      = tuple(config.model.hiera.get("mask_unit_attn", [True, True, False, False]))
dim_mul             = config.model.hiera.get("dim_mul", 2.0)
head_mul            = config.model.hiera.get("head_mul", 2.0)
patch_kernel        = tuple(config.model.hiera.get("patch_kernel", [7, 7]))
patch_stride        = tuple(config.model.hiera.get("patch_stride", [4, 4]))
patch_padding       = tuple(config.model.hiera.get("patch_padding", [3, 3]))
mlp_ratio           = config.model.hiera.get("mlp_ratio", 4.0)
drop_path_rate      = config.model.hiera.get("drop_path_rate", 0.0)

# Handle norm_layer - convert string to actual layer class
norm_layer_str      = config.model.hiera.get("norm_layer", "LayerNorm")
norm_layer          = partial(getattr(nn, norm_layer_str), eps=1e-6) if isinstance(norm_layer_str, str) else norm_layer_str
head_dropout        = config.model.hiera.get("head_dropout", 0.0)
head_init_scale     = config.model.hiera.get("head_init_scale", 0.001)
sep_pos_embed       = config.model.hiera.get("sep_pos_embed", False)

# -- Loss
grad_accum_steps = max(int(config.loss.grad_accum_steps), 1)
focal_alpha      = config.loss.focal.alpha
focal_gamma      = config.loss.focal.gamma

# -- Optimizer
lr           = float(config.optim.lr)
weight_decay = float(config.optim.weight_decay)
adam_beta1   = float(config.optim.beta1)
adam_beta2   = float(config.optim.beta2)
adam_fused   = float(config.optim.fused)
grad_clip    = float(config.optim.grad_clip)

# -- Scheduler
warmup_steps = config.lr_scheduler.warmup_steps
total_steps = config.lr_scheduler.total_steps
min_lr = float(config.lr_scheduler.min_lr)
scheduler_update_steps = config.lr_scheduler.scheduler_update_steps

# -- Distributed envs
dist_backend           = config.dist.backend
uses_unique_world_seed = config.dist.uses_unique_world_seed
dist_dtype             = config.dist.dtype
cpu_offload            = config.dist.cpu_offload

# -- Logging
drc_log        = config.logging.directory
fl_log_prefix  = config.logging.prefix
log_level      = config.logging.level

# -- Misc
max_eval_iter      = config.misc.max_eval_iter
max_eval_retry     = config.misc.max_eval_retry
compiles_model     = config.misc.compiles_model
data_dump_on       = config.misc.get("data_dump_on", False)
cpu_only           = config.misc.get("cpu_only", False)
peak_flops_per_sec = config.misc.peak_flops_per_sec
monitors_dynamics  = config.misc.monitors_dynamics
sharding_stage     = config.misc.sharding_stage


# ----------------------------------------------------------------------- #
#  MISC FEATURES
# ----------------------------------------------------------------------- #
# Register signal handlers for graceful shutdown
register_handlers()

# ----------------------------------------------------------------------- #
#  DIST SETUP
# ----------------------------------------------------------------------- #
# Initialize distributed environment using utility function
dist_rank, dist_local_rank, dist_world_size, uses_dist, device = dist_setup(
    cpu_only=cpu_only,
    device_per_node=None,  # Auto-detect available GPUs
    dist_backend=dist_backend
)

# --- Set up performance utility
memmax = MemoryMaximizer() if dist_local_rank == 0 else None

# --- Seed setup
seed_offset = dist_rank if uses_unique_world_seed else 0

# ----------------------------------------------------------------------- #
#  FSDP SETUP
# ----------------------------------------------------------------------- #
# -- FSDP policy using utilities
sharding_strategy = set_sharding_strategy(sharding_stage)
auto_wrap_policy = shard_layers({HieraBlock, MaskedAutoencoderHiera})
backward_prefetch_policy = backward_prefetch()

# ----------------------------------------------------------------------- #
#  TF32 support
# ----------------------------------------------------------------------- #
# Ampere architecture (capability_major = 8) is required.
if device != 'cpu':
    capability_major, capability_minor = torch.cuda.get_device_capability(device)
    if capability_major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if dist_rank == 0:
            logger.info("TF32 enabled on matmul and cuDNN operations.")


# ----------------------------------------------------------------------- #
#  LOGGING
# ----------------------------------------------------------------------- #
# Set up universal distributed logging
timestamp = logger_utils.setup_distributed_logging(
    prefix=fl_log_prefix,
    log_dir=drc_log,
    level=log_level,
)

if dist_rank == 0:
    # Convert OmegaConf to yaml formatted string...
    config_yaml = OmegaConf.to_yaml(config)

    # Log the config...
    logger.info(config_yaml)

# ----------------------------------------------------------------------- #
#  DATASET
# ----------------------------------------------------------------------- #
logger.info('Configuring dataset...')
# -- Seeding
base_seed  = 0
world_seed = base_seed + seed_offset
set_seed(world_seed)

# -- Set up transformation
class NoTransform:
    def __call__(self, x, **kwargs):
        return x

merges_batch_patch_dims = uses_polar_center_crop
pre_transforms = (
    Pad(H_pad, W_pad) if uses_pad else NoTransform(),
    PolarCenterCrop(
        Hv=Hv,
        Wv=Wv,
        sigma=sigma,
        num_crop=num_crop,
    ) if uses_polar_center_crop else NoTransform(),
    MergeBatchPatchDims() if merges_batch_patch_dims else NoTransform(),
)
transforms = None

# -- Set up training set
dataset_train_config = PeakNetDatasetConfig(
    path_csv=path_dataset_train,
    transforms=pre_transforms,
    buffer_size=cache_size,
    dist_rank=dist_rank,
    dist_world_size=dist_world_size,
    device=str(device),
    dtype=None,
    uses_norm=True,
    scales_variance=True,
    perfs_runtime=False,
)
dataset_train = PeakNetDataset(dataset_train_config)

# -- Set up eval set
# --- For training loss
dataset_eval_train = PeakNetDataset(dataset_train_config)

# --- For val loss
dataset_eval_val_config = PeakNetDatasetConfig(
    path_csv=path_dataset_eval,
    transforms=pre_transforms,
    buffer_size=cache_size,
    dist_rank=dist_rank,
    dist_world_size=dist_world_size,
    device=str(device),
    dtype=None,
    uses_norm=True,
    scales_variance=True,
    perfs_runtime=False,
)
dataset_eval_val = PeakNetDataset(dataset_eval_val_config)

# -- Custom collate to merge patch and batch dimension using concatenation
custom_collate = None

# ----------------------------------------------------------------------- #
#  TIMESTAMP
# ----------------------------------------------------------------------- #
# Generate fresh timestamp for each run session (like train.fsdp.py)
if dist_rank == 0:
    run_timestamp = time.strftime("%Y_%m%d_%H%M")
else:
    run_timestamp = None
# Broadcast timestamp to all ranks for consistency
run_timestamp = broadcast_timestamp_to_all_ranks(run_timestamp, dist_rank, uses_dist)

# Simple config-driven resumption (like train.fsdp.py)
from_resume = path_chkpt_prev is not None

if dist_rank == 0:
    logger.info(f"Run timestamp: {run_timestamp}")
    if from_resume:
        logger.info(f"Will resume from: {path_chkpt_prev}")

# ----------------------------------------------------------------------- #
#  CHECKPOINT PRE FSDP
# ----------------------------------------------------------------------- #
# Determine FSDP usage based on sharding strategy and distributed setup
uses_fsdp = uses_dist and sharding_stage != 'zero0'  # Only use FSDP in distributed mode
checkpointer = init_checkpointer(
    state_dict_type=state_dict_type if uses_dist else "full",  # Force "full" for single GPU
    uses_fsdp=uses_fsdp,
    offload_to_cpu=chkpt_offload_to_cpu,
    rank0_only=chkpt_rank0_only
)

# ----------------------------------------------------------------------- #
#  MODEL
# ----------------------------------------------------------------------- #
# -- Config the model with explicit parameters
logger.info('Configuring model...')
model = HieraSegmentation(
    # Segmentation-specific parameters
    num_classes=num_classes,
    decoder_embed_dim=decoder_embed_dim,
    decoder_depth=decoder_depth,
    decoder_num_heads=decoder_num_heads,

    # Core Hiera architecture parameters (passed via **kwargs to parent classes)
    input_size=input_size,
    in_chans=in_chans,
    embed_dim=embed_dim,
    num_heads=num_heads,
    stages=stages,
    q_pool=q_pool,
    q_stride=q_stride,
    mask_unit_size=mask_unit_size,
    mask_unit_attn=mask_unit_attn,
    dim_mul=dim_mul,
    head_mul=head_mul,
    patch_kernel=patch_kernel,
    patch_stride=patch_stride,
    patch_padding=patch_padding,
    mlp_ratio=mlp_ratio,
    drop_path_rate=drop_path_rate,
    norm_layer=norm_layer,
    head_dropout=head_dropout,
    head_init_scale=head_init_scale,
    sep_pos_embed=sep_pos_embed,
)

# Initialize weights (HieraSegmentation handles this in its constructor)
if hasattr(model, 'init_weights') and callable(model.init_weights):
    model.init_weights()
## if not uses_dist: model.to(device)

# !! Make all params trainable, a workaround for pytorch 2.0.1
torch_version = torch.__version__
torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
if version.parse(torch_version) <= version.parse("2.0.1"):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True

if dist_rank == 0:
    logger.info(f"[MODEL SETUP] {sum(p.numel() for p in model.parameters())/1e6} M parameters.")

if from_resume:
    if hasattr(checkpointer, 'pre_dp_load'):
        checkpointer.pre_dp_load(dist_rank, model, path_chkpt_prev)

# -- Mixed precision
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dist_dtype]
mixed_precision = MixedPrecision(
    param_dtype  = mixed_precision_dtype,
    reduce_dtype = mixed_precision_dtype,
    buffer_dtype = mixed_precision_dtype,
)

# --- Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = mixed_precision_dtype)

# --- GradScaler
# If enabled = False scaler is a no-op
if uses_fsdp:
    scaler = ShardedGradScaler(enabled=(dist_dtype == 'float16'))
else:
    scaler = torch.amp.GradScaler('cuda', enabled=(dist_dtype == 'float16'))

# -- Compile the model
if compiles_model:
    logger.info("[MODEL SETUP] Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

# -- Wrapping the model with FSDP or DDP based on sharding strategy
if uses_dist:
    # Convert BatchNorm to SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if uses_fsdp:
        # Wrap with FSDP
        model = FSDP(
            model,
            auto_wrap_policy  = auto_wrap_policy,
            mixed_precision   = mixed_precision,
            backward_prefetch = backward_prefetch_policy,
            forward_prefetch  = True,
            sharding_strategy = sharding_strategy,
            limit_all_gathers = True,
            cpu_offload       = None if cpu_offload is None else CPUOffload(offload_params=cpu_offload),
            use_orig_params   = False,
            device_id         = device,
        )
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"[MODEL SETUP] FSDP sharded parameter count: {param_count*1e-6} M.")
    else:
        # Wrap with DDP (when sharding_stage = 'zero0')
        model.to(device)
        model = DDP(model, device_ids=[dist_local_rank])
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"[MODEL SETUP] DDP parameter count: {param_count*1e-6} M.")

    dist.barrier()
else:
    # Single GPU - just move to device
    model.to(device)

# -- Optional grad sync off (to allow grad accumulation)
# Works with both FSDP and DDP
grad_sync_context = lambda enables_sync: nullcontext() if enables_sync or not uses_dist else model.no_sync()

# -- Apply activation checkpointing using utility function
act_chkpt(model, (HieraBlock,))

if dist_rank == 0:
    logger.info(f"Current timestamp: {timestamp}")

# ----------------------------------------------------------------------- #
#  CRITERION (LOSS)
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring criterion...')
criterion = CategoricalFocalLoss(
    alpha       = focal_alpha,
    gamma       = focal_gamma,
    num_classes = num_classes,
)


# ----------------------------------------------------------------------- #
#  OPTIMIZER AND SCHEDULER
# ----------------------------------------------------------------------- #
logger.info('Configuring optimizer...')
param_iter = model.parameters()
optim_arg_dict = dict(
    lr           = lr,
    weight_decay = weight_decay,
    betas        = (adam_beta1, adam_beta2),
)
if 'fused' in inspect.signature(optim.AdamW).parameters:
    optim_arg_dict['fused'] = adam_fused
optimizer = optim.AdamW(param_iter, **optim_arg_dict)
scheduler = CosineLRScheduler(optimizer = optimizer,
                              warmup_iterations = warmup_steps,
                              total_iterations  = total_steps,
                              min_lr = min_lr)


# ----------------------------------------------------------------------- #
#  CHECKPOINT POST FSDP
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring model, optim, scheduler, training state checkpoint...')
# -- Set init training state dict
starting_step = 0
loss_min = float('inf')
step_state = dict(
    step        = starting_step,
    loss_min    = loss_min,
    timestamp   = run_timestamp,
)

# -- Optional resumption
if from_resume:
    if hasattr(checkpointer, 'post_dp_load'):
        # Load model, optimizer, scheduler, and step_state from checkpoint
        checkpointer.post_dp_load(dist_rank, model, optimizer, scheduler, step_state, path_chkpt_prev)

        # Extract training state from loaded checkpoint data (not filename!)
        starting_step = step_state.get("step", 0) + 1  # Resume from next step
        loss_min = step_state.get("loss_min", loss_min)

        # Log resumption info
        logger_utils.log_on_all_ranks(logger, f"[RESUMPTION] Loading from checkpoint -- {path_chkpt_prev}", "info")
        logger_utils.log_on_all_ranks(logger, f"[RESUMPTION] Resuming from step: {starting_step-1}-->{starting_step} , loss_min: {loss_min}", "info")


# ----------------------------------------------------------------------- #
#  Monitoring training dynamics
# ----------------------------------------------------------------------- #
if monitors_dynamics:
    # Use GELU activation for Hiera (hardcoded since it doesn't use transformers ACT2CLS)
    modules_to_monitor = (nn.GELU, )
    act_monitor = ActivationMonitor(model, modules_to_monitor)
    act_monitor.add_hooks()
else:
    act_monitor = None

# ----------------------------------------------------------------------- #
#  HELPER
# ----------------------------------------------------------------------- #
def get_num_params(model):
    return sum(p.numel() for p in model.parameters())

# Calculate FLOPS per token for the model
dummy_shape = 1, in_chans, input_size[0], input_size[1]  # Use actual model input size
# For Hiera, use patch_stride instead of patch_size
# Assuming patch_stride is a tuple like (4, 4), we use the first element
hiera_patch_size = patch_stride[0] if isinstance(patch_stride, (tuple, list)) else patch_stride
num_flops_per_token = estimate_flops_per_token(model, dummy_shape, hiera_patch_size, count_multiply_add_as=2, includes_backward=True, device=device)


# ----------------------------------------------------------------------- #
#  TRAINING LOOP
# ----------------------------------------------------------------------- #
batch_input_shape = None
logger.info('Ready for the training loop...')

# Initialize step counter from resumption
step_counter = starting_step

# Create infinite cycling dataloader
infinite_dataloader, sampler, batches_per_epoch = create_infinite_dataloader(
    dataset=dataset_train,
    base_seed=base_seed,
    drop_last_in_sampler=drop_last_in_sampler,
    drop_last_in_loader=drop_last_in_loader,
    uses_dist=uses_dist,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
)

if dist_rank == 0:
    logger.info(f"Training for {total_steps} total steps (starting from step {starting_step})")
    logger.info(f"[TRAINING] Run timestamp: {run_timestamp}")
    logger.info(f"[TRAINING] Checkpoint saving every {chkpt_saving_steps} steps")
    logger.info(f"[TRAINING] Scheduler update every {scheduler_update_steps} steps")
    logger.info(f"[TRAINING] Gradient accumulation steps: {grad_accum_steps}")
    logger.info(f"[TRAINING] Batches per epoch: {batches_per_epoch}")

try:
    model.train()

    # Initialize gradient accumulation state
    grad_nosync_counter = 0
    batch_idx = 0

    if dist_rank == 0:
        logger.info(f"[TRAINING] Starting step-based training loop from step {starting_step}")

    # [PERFORMANCE] Start memory monitoring
    if dist_local_rank == 0:
        memmax.start()

    # Step-based training loop
    for batch_data in infinite_dataloader:
        if step_counter >= total_steps:
            break

        if batch_data is None:
            continue

        # Process batch data
        batch_data = rearrange(batch_data, 'n b c h w -> (n b) c h w')  # (2*B, C, H, W)
        batch_data = batch_data.to(device, non_blocking=True, dtype=mixed_precision_dtype)

        # Apply transforms
        if transforms is not None:
            for trans in transforms:
                batch_data = trans(batch_data)

        # Split input and target
        batch_input, batch_target = rearrange(batch_data, '(n b) c h w -> n b c h w', n=2)

        # Binarize labels
        if transforms is not None:
            batch_target = batch_target > 0.5

        # Determine if gradient sync is required for this batch
        # Detect last batch of current epoch cycle (same logic as train.fsdp.py)
        is_last_batch_of_epoch = (batch_idx % batches_per_epoch) == (batches_per_epoch - 1)
        is_grad_sync_required = is_last_batch_of_epoch or is_action_due(grad_nosync_counter, grad_accum_steps)

        # Forward and backward pass with proper gradient accumulation
        with grad_sync_context(is_grad_sync_required):
            with autocast_context:
                batch_output = model(batch_input)
                loss = criterion(batch_output, batch_target)
                loss = loss.mean()
                loss = loss / grad_accum_steps  # Scale for gradient accumulation

            # Backward pass
            scaler.scale(loss).backward()

        # Increment gradient accumulation counter and batch index
        grad_nosync_counter += 1
        batch_idx += 1

        # Only update parameters when gradient sync is required
        if is_grad_sync_required:
            # Log gradient sync event for verification
            if dist_rank == 0:
                sync_reason = "epoch_boundary" if is_last_batch_of_epoch else "grad_accum_complete"
                logger.info(f"[GRADIENT SYNC] Triggered by {sync_reason} at batch_idx={batch_idx}, grad_nosync_counter={grad_nosync_counter}")

            # Gradient clipping and optimizer step
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip) \
                            if (not uses_dist) or sharding_strategy == ShardingStrategy.NO_SHARD \
                            else \
                            model.clip_grad_norm_(grad_clip)

            # Update parameters
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Reset gradient accumulation counter and batch index
            grad_nosync_counter = 0

            # Increment step counter only after parameter update
            step_counter += 1

            # [PERFORMANCE] Update memory monitoring
            if dist_local_rank == 0:
                memmax.update()

            # Log step counter increment for verification
            if dist_rank == 0:
                logger.info(f"[STEP] Step counter incremented to {step_counter} after parameter update")
                logger.info(f"[LOSS] Step {step_counter}: loss={loss:.10f}")

            # Update scheduler
            if is_action_due(step_counter, scheduler_update_steps):
                if dist_rank == 0:
                    logger.info(f"[SCHEDULER] Scheduler update triggered at step {step_counter}")
                scheduler.step()
                if dist_rank == 0:
                    current_lrs = scheduler.get_lr()
                    current_lrs_msg = ",".join(f"{lr}" for lr in current_lrs)
                    logger.info(f"[SCHEDULER] lr updated to {current_lrs_msg} at step {step_counter}")

            # Logging (only on main process)
            if dist_rank == 0 and step_counter % 10 == 0:
                current_lrs = scheduler.get_lr()
                logger.info(f"Step {step_counter}: loss={loss:.6f}, lr={current_lrs[0]:.2e}")

            # Enhanced logging for stress testing (every step near checkpoints)
            if step_counter % chkpt_saving_steps == 0 or step_counter % chkpt_saving_steps == 1:
                if dist_rank == 0:
                    current_lrs = scheduler.get_lr()
                    logger.info(f"[STRESS TEST] Step {step_counter}: loss={loss:.8f}, lr={current_lrs[0]:.2e} (checkpoint boundary)")

            # Checkpointing
            if is_action_due(step_counter, chkpt_saving_steps):
                if dist_rank == 0:
                    logger.info(f"[CHECKPOINT] Checkpoint triggered at step {step_counter}")
                    logger.info(f"[CHECKPOINT] Expected checkpoint name: {fl_chkpt_prefix}_{run_timestamp}_step_{step_counter}")

                # Evaluation
                model.eval()

                with torch.no_grad():
                    # Training set
                    eval_dataloader_train, sampler_eval = wrap_with_torch_dataloader(
                        dataset=dataset_eval_train,
                        base_seed=base_seed,
                        drop_last_in_sampler=drop_last_in_sampler,
                        drop_last_in_loader=drop_last_in_loader,
                        uses_dist=uses_dist,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        custom_collate=custom_collate,
                        pin_memory=pin_memory,
                        prefetch_factor=prefetch_factor,
                        epoch=0,  # Fixed epoch for eval
                        is_eval=True,
                    )

                    if uses_dist and sampler_eval is not None:
                        sampler_eval.set_epoch(0)

                    train_eval_loss = estimate_loss(
                        eval_dataloader_train,
                        model,
                        criterion,
                        autocast_context,
                        max_iter=max_eval_iter,
                        desc='(training set)',
                        device=device,
                        dummy_input_shape=batch_input_shape,
                        mixed_precision_dtype=mixed_precision_dtype,
                        transforms=transforms,
                        uses_dist=uses_dist,
                        dist_rank=dist_rank,
                        dist_world_size=dist_world_size,
                        data_dump_on=data_dump_on,
                    )

                    # Validation set
                    eval_dataloader_val, sampler_eval = wrap_with_torch_dataloader(
                        dataset=dataset_eval_val,
                        base_seed=base_seed,
                        drop_last_in_sampler=drop_last_in_sampler,
                        drop_last_in_loader=drop_last_in_loader,
                        uses_dist=uses_dist,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        custom_collate=custom_collate,
                        pin_memory=pin_memory,
                        prefetch_factor=prefetch_factor,
                        epoch=0,  # Fixed epoch for eval
                        is_eval=True,
                    )

                    if uses_dist and sampler_eval is not None:
                        sampler_eval.set_epoch(0)

                    val_eval_loss = estimate_loss(
                        eval_dataloader_val,
                        model,
                        criterion,
                        autocast_context,
                        max_iter=max_eval_iter,
                        desc='(validation set)',
                        device=device,
                        dummy_input_shape=batch_input_shape,
                        mixed_precision_dtype=mixed_precision_dtype,
                        transforms=transforms,
                        uses_dist=uses_dist,
                        dist_rank=dist_rank,
                        dist_world_size=dist_world_size,
                        data_dump_on=data_dump_on,
                    )

                # Process evaluation results
                # Use validation loss for model selection if available, otherwise use training loss
                if not torch.isnan(val_eval_loss) and not torch.isinf(val_eval_loss):
                    eval_loss = val_eval_loss
                else:
                    eval_loss = train_eval_loss if not torch.isnan(train_eval_loss) and not torch.isinf(train_eval_loss) else float('inf')

                if dist_rank == 0:
                    # Log both losses if both are valid
                    if (not torch.isnan(train_eval_loss) and not torch.isinf(train_eval_loss) and 
                        not torch.isnan(val_eval_loss) and not torch.isinf(val_eval_loss)):
                        logger.info(f"Step {step_counter}: train_eval_loss={train_eval_loss:.6f}, val_eval_loss={val_eval_loss:.6f}")
                    elif not torch.isnan(train_eval_loss) and not torch.isinf(train_eval_loss):
                        logger.info(f"Step {step_counter}: train_eval_loss={train_eval_loss:.6f}")
                    elif not torch.isnan(val_eval_loss) and not torch.isinf(val_eval_loss):
                        logger.info(f"Step {step_counter}: val_eval_loss={val_eval_loss:.6f}")

                # Save best model checkpoint if evaluation loss improved
                if eval_loss < loss_min and eval_loss != float('inf'):
                    loss_min = eval_loss

                    # Update step state
                    step_state["step"] = step_counter
                    step_state["loss_min"] = loss_min
                    step_state["timestamp"] = run_timestamp

                    best_output_dir = f"{fl_chkpt_prefix}_{run_timestamp}_best_step_{step_counter}"
                    best_output_path = os.path.join(dir_root_chkpt, best_output_dir)

                    if dist_rank == 0:
                        logger.info(f"[CHECKPOINT] Saving BEST checkpoint: {best_output_dir}")

                    checkpointer.save(dist_rank, model, optimizer, scheduler, step_state, best_output_path)

                    if dist_rank == 0:
                        logger.info(f"[CHECKPOINT] BEST checkpoint saved successfully: {best_output_dir} (val_loss={eval_loss:.6f})")


                model.train()  # Back to training mode

            # Preemptive checkpointing (for HPC time limits)
            if preempt_metadata_path and is_action_due(step_counter, preempt_chkpt_saving_steps):
                step_state["step"] = step_counter
                step_state["loss_min"] = loss_min
                step_state["timestamp"] = run_timestamp

                preempt_output_dir = f"{fl_chkpt_prefix}_{run_timestamp}.preempt"
                preempt_output_path = os.path.join(dir_root_chkpt, preempt_output_dir)

                if dist_rank == 0:
                    logger.info(f"[CHECKPOINT] Saving PREEMPTIVE checkpoint: {preempt_output_dir}")

                checkpointer.save(dist_rank, model, optimizer, scheduler, step_state, preempt_output_path)

                # Write metadata file with checkpoint path
                if dist_rank == 0:
                    with open(preempt_metadata_path, "w") as f:
                        f.write(preempt_output_path)
                    logger.info(f"[CHECKPOINT] PREEMPTIVE checkpoint saved: {preempt_output_dir}")
                    logger.info(f"[CHECKPOINT] Preemptive metadata written to: {preempt_metadata_path}")

    # Training completed
    if dist_rank == 0:
        logger.info(f"[TRAINING] Training completed after {step_counter} steps")
        logger.info(f"[TRAINING] Final loss_min: {loss_min}")

except KeyboardInterrupt:
    logger_utils.log_on_all_ranks(logger, "Training was interrupted!", "error")
except Exception as e:
    tb = traceback.format_exc()
    logger_utils.log_on_all_ranks(logger, f"Error occurred: {e}\nTraceback: {tb}", "error")
finally:
    # [PERFORMANCE] Stop memory monitoring
    if dist_local_rank == 0 and memmax is not None:
        memmax.stop()

    # Clean up hooks
    if monitors_dynamics and act_monitor is not None:
        act_monitor.remove_hooks()

    # Ensure that the process group is always destroyed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
