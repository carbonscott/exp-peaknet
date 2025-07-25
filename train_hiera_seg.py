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
from peaknet.datasets.segmented_zarr_dataset import (
    SegmentedPeakNetDatasetConfig,
    SegmentedPeakNetDataset,
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
from peaknet.utils.data        import wrap_with_torch_dataloader
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
from peaknet.utils.eval import estimate_loss, is_last_batch

# -- Imports for monitoring training dynamics
# Note: Removed transformers.activations import as not needed for Hiera

# -- Torch specific imports
import torch
import torch.nn as nn
import torch.optim as optim

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
dir_root_chkpt                  = config.checkpoint.directory
fl_chkpt_prefix                 = config.checkpoint.prefix
path_chkpt_prev                 = config.checkpoint.get("path_chkpt_prev", None)
chkpt_saving_iterations         = config.checkpoint.chkpt_saving_iterations
preempt_metadata_path           = config.checkpoint.get("preempt_metadata_path", os.environ.get('PREEMPT_METADATA_PATH', None))
preempt_chkpt_saving_iterations = config.checkpoint.preempt_chkpt_saving_iterations
state_dict_type                 = config.checkpoint.state_dict_type
chkpt_offload_to_cpu            = config.checkpoint.offload_to_cpu
chkpt_rank0_only                = config.checkpoint.rank0_only

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
num_patch              = config.dataset.transforms.num_patch
size_patch             = config.dataset.transforms.size_patch
frac_shift_max         = config.dataset.transforms.frac_shift_max
angle_max              = config.dataset.transforms.angle_max
var_size_patch         = config.dataset.transforms.var_size_patch
patch_size             = config.dataset.transforms.patch_size
stride                 = config.dataset.transforms.stride
detector_norm_params   = config.dataset.transforms.norm
sampling_fraction      = config.dataset.transforms.get("sampling_fraction", None)
H_pad                  = config.dataset.transforms.H_pad
W_pad                  = config.dataset.transforms.W_pad
Hv                     = config.dataset.transforms.Hv
Wv                     = config.dataset.transforms.Wv
sigma                  = config.dataset.transforms.sigma
num_crop               = config.dataset.transforms.num_crop
uses_pad               = config.dataset.transforms.set.pad
uses_random_patch      = config.dataset.transforms.set.random_patch
uses_random_rotate     = config.dataset.transforms.set.random_rotate
uses_random_shift      = config.dataset.transforms.set.random_shift
uses_polar_center_crop = config.dataset.transforms.set.polar_center_crop
uses_batch_sampler     = config.dataset.transforms.set.batch_sampler

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
patience                    = config.lr_scheduler.patience
warmup_iterations           = config.lr_scheduler.warmup_iterations
total_iterations            = config.lr_scheduler.total_iterations
min_lr                      = float(config.lr_scheduler.min_lr)
scheduler_update_iterations = config.lr_scheduler.scheduler_update_iterations

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
max_epochs         = config.misc.max_epochs
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
logger.debug('Configuring dataset...')
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
)

transforms = (
    PolarCenterCrop(
        Hv       = Hv,
        Wv       = Wv,
        sigma    = sigma,
        num_crop = num_crop,
    ) if uses_polar_center_crop else NoTransform(),
    MergeBatchPatchDims() if merges_batch_patch_dims else NoTransform(),
    BatchSampler(sampling_fraction) if uses_batch_sampler else NoTransform(),
    RandomPatch(
        num_patch    = num_patch,
        H_patch      = size_patch,
        W_patch      = size_patch,
        var_H_patch  = var_size_patch,
        var_W_patch  = var_size_patch,
        returns_mask = False,
    ) if uses_random_patch  else NoTransform(),
    RandomRotate(angle_max) if uses_random_rotate else NoTransform(),
    RandomShift(
        frac_y_shift_max = frac_shift_max,
        frac_x_shift_max = frac_shift_max,
    ) if uses_random_shift  else NoTransform(),
)

# -- Set up training set
dataset_train_config = SegmentedPeakNetDatasetConfig(
    path_csv        = path_dataset_train,
    seg_size        = seg_size,
    transforms      = pre_transforms,
    buffer_size     = 1,
    dist_rank       = dist_rank,
    dist_world_size = dist_world_size,
    device          = device,
    dtype           = None,
    perfs_runtime   = False,
)
dataset_train = SegmentedPeakNetDataset(dataset_train_config)

# -- Set up eval set
# --- For training loss
dataset_eval_train = SegmentedPeakNetDataset(dataset_train_config)

# --- For val loss
dataset_eval_val_config = SegmentedPeakNetDatasetConfig(
    path_csv        = path_dataset_eval,
    seg_size        = seg_size,
    transforms      = pre_transforms,
    buffer_size     = 1,
    dist_rank       = dist_rank,
    dist_world_size = dist_world_size,
    device          = device,
    dtype           = None,
    perfs_runtime   = False,
)
dataset_eval_val = SegmentedPeakNetDataset(dataset_eval_val_config)

# -- Custom collate to merge patch and batch dimension using concatenation
## custom_collate = lambda batch: torch.cat(batch, dim = 0)  # batch of [N, C, H, W] -> [B * N, C, H, W]
## def custom_collate(batch):
##     batch_filtered = [x for x in batch if x is not None]
##     return torch.cat(batch_filtered, dim = 0) if len(batch_filtered) else None
custom_collate = None

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
from_resume = path_chkpt_prev is not None

# ----------------------------------------------------------------------- #
#  MODEL
# ----------------------------------------------------------------------- #
logger.debug('Configuring model...')
# -- Config the model with explicit parameters
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
if not uses_dist: model.to(device)

# !! Make all params trainable, a workaround for pytorch 2.0.1
torch_version = torch.__version__
torch_version = torch_version[:torch_version.find("+") if "+" in torch_version else None]
if version.parse(torch_version) <= version.parse("2.0.1"):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True

if dist_rank == 0:
    logger.debug(f"{sum(p.numel() for p in model.parameters())/1e6} M parameters.")

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
    logger.debug("Compiling the model...")
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
        logger.debug(f"FSDP sharded parameter count: {param_count*1e-6} M.")
    else:
        # Wrap with DDP (when sharding_stage = 'zero0')
        model.to(device)
        model = DDP(model, device_ids=[dist_local_rank])
        param_count = sum(p.numel() for p in model.parameters())
        logger.debug(f"DDP parameter count: {param_count*1e-6} M.")

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
    logger.debug(f"Current timestamp: {timestamp}")

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
logger.debug('Configuring optimizer...')
param_iter = model.parameters()
optim_arg_dict = dict(
    lr           = lr,
    weight_decay = weight_decay,
    betas        = (adam_beta1, adam_beta2),
)
if 'fused' in inspect.signature(optim.AdamW).parameters:
    optim_arg_dict['fused'] = adam_fused
optimizer = optim.AdamW(param_iter, **optim_arg_dict)
scheduler = CosineLRScheduler(optimizer         = optimizer,
                              warmup_iterations = warmup_iterations,
                              total_iterations  = total_iterations,
                              min_lr            = min_lr)


# ----------------------------------------------------------------------- #
#  CHECKPOINT POST FSDP
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring model, optim, scheduler, training state checkpoint...')
# -- Set init training state dict
loss_min = float('inf')
iter_state = dict(
    epoch     = 0,
    seg       = 0,
    start_idx = dataset_train.start_idx,
    end_idx   = dataset_train.end_idx,
    loss_min  = loss_min,
)

# -- Optional resumption
last_epoch = 0
last_seg   = -1
if from_resume:
    if hasattr(checkpointer, 'post_dp_load'):
        # Optimizer, scheduler are loaded
        checkpointer.post_dp_load(dist_rank, model, optimizer, scheduler, iter_state, path_chkpt_prev)

        # Training state
        last_epoch = iter_state.get("epoch")
        last_seg   = iter_state.get("seg")
        loss_min   = iter_state.get("loss_min")

        # Log checkpoint loading with smart utility for visibility
        logger_utils.log_on_all_ranks(logger, f"Loading from checkpoint -- {path_chkpt_prev}.", "info")
        logger_utils.log_on_all_ranks(logger, f"PREV - last_epoch {last_epoch}, last_seg {iter_state.get('start_idx')}-{iter_state.get('end_idx')}, loss_min = {loss_min}", "info")


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
logger.debug('Ready for training loop...')
iteration_counter = 0  # One iteration is one param update after one or a few forward/backward pass
try:
    # -- Loop over epochs
    # Only increment starting epoch if current epoch was fully completed
    for epoch in tqdm.tqdm(range(max_epochs), desc = f'[RANK {dist_rank}] Epoch'):
        # Skip epochs up to, but not including the last_epoch
        if epoch < last_epoch: continue

        # Reset dataset in a new epoch???
        if not from_resume:
            dataset_train.reset()

        # Otherwise, update the dataset index according to the training state
        else:
            # Update the dataset status
            dataset_train.start_idx = iter_state.get("start_idx")
            dataset_train.end_idx   = iter_state.get("end_idx")

        # -- Loop over dataset segments
        for seg in tqdm.tqdm(range(dataset_train.num_seg), desc = f'[RANK {dist_rank}] Segment'):
            # Skip previous segments up to and including the last_seg
            if epoch == last_epoch and seg <= last_seg:
                continue

            # Switch to training state
            model.train()

            # Prepare training on one segment (iteration)
            # Set next segment or break the loop when having no next segment
            requires_reset = dataset_train.set_start_idx(dataset_train.end_idx)
            if requires_reset:
                break

            # [PERFORMANCE]
            if dist_local_rank == 0:
                memmax.start()

            if dist_rank == 0:
                logger.info(f"Working on segment: {dataset_train.start_idx}:{dataset_train.end_idx}")

            # Create training dataloader using utility function
            dataloader, sampler = wrap_with_torch_dataloader(
                dataset=dataset_train,
                base_seed=base_seed,
                drop_last_in_sampler=drop_last_in_sampler,
                drop_last_in_loader=drop_last_in_loader,
                uses_dist=uses_dist,
                batch_size=batch_size,
                num_workers=num_workers,
                custom_collate=custom_collate,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                epoch=epoch,
                is_eval=False,
            )

            # Set epoch for distributed sampler (handled by utility)
            if uses_dist and sampler is not None:
                sampler.set_epoch(epoch)

            # -- Loop over mini batches
            # --- Set up helper variables for gradient accum and reporting
            # Set up gradient accumulation helper variables
            grad_nosync_counter         = 0
            num_batches                 = len(dataloader)
            num_remainder_batches       = num_batches % grad_accum_steps
            start_idx_remainder_batches = num_batches - num_remainder_batches  # e.g. total=102, steps=5, idx = 102 - 102%5 = 100

            # Aggregate the loss and number of processed tokens during each gradient accumulation
            total_loss       = torch.tensor(0.0, device = device)
            total_num_tokens = torch.tensor(0.0, device = device)

            # Set a timer flag
            starts_timer = True

            # --- Mini batch loop
            logger.debug(f"Start processing {len(dataloader)} batches at epoch {epoch}, seg {seg}.")
            for batch_idx, batch_data in tqdm.tqdm(
                enumerate(dataloader),
                total = num_batches,
                desc  = f'[RANK {dist_rank}] Mini batch',
            ):
                # Start timer???
                if starts_timer:
                    t_start = time.monotonic()
                    starts_timer = False

                # ---- Forward/Backward during an iteration
                # Create dummy data for a None batch
                # FIXME: Better data cleaning will eliminate None batch
                if batch_data is None:
                    logger.debug(f"Found None batch at batch idx {batch_idx}.  Creating a dummy input!!!")
                    batch_input  = torch.zeros(batch_input_shape, dtype = mixed_precision_dtype)
                    batch_target = torch.zeros(batch_input_shape, dtype = mixed_precision_dtype)
                    batch_data = (batch_input, batch_target)

                # Concat data to perform the identical transform on input and target
                batch_data = torch.cat(batch_data, dim = 0)    # (2*B, C, H, W)
                batch_data = batch_data.to(device, non_blocking = True, dtype = mixed_precision_dtype)

                # Optional transform
                if transforms is not None:
                    for enum_idx, trans in enumerate(transforms):
                        batch_data = trans(batch_data)

                # Unpack vars
                current_batch_size = batch_data.size(0) // 2
                batch_input  = batch_data[                  :current_batch_size]
                batch_target = batch_data[current_batch_size:                  ]

                # Optionally binarize the label
                if transforms is not None:
                    batch_target = batch_target > 0.5

                # Specify the effective grad accum steps
                real_grad_accum_steps = grad_accum_steps if batch_idx < start_idx_remainder_batches else num_remainder_batches

                # Conditionally turn off grad sync for grad accumulation to simulate a larger batch unless the sync is due or the last batch
                # Refer to https://github.com/pytorch/pytorch/blob/6c4f43f82675b5fcfe8cf3e5983d0c0f326408aa/test/distributed/fsdp/test_fsdp_grad_acc.py#L180
                is_grad_sync_required = is_last_batch(batch_idx, len(dataloader)) or is_action_due(grad_nosync_counter, grad_accum_steps)
                with grad_sync_context(is_grad_sync_required):
                    # Forward
                    with autocast_context:
                        batch_output = model(batch_input)
                        loss = criterion(batch_output, batch_target)
                        loss = loss.mean()
                        loss = loss / real_grad_accum_steps  # scale the loss to account for gradient accumulation
                        logger.debug(f"loss = {loss}")

                    # Accumulate loss
                    total_loss += loss.detach()

                    # Accumulate number of tokens processed
                    total_numel = batch_input.numel()  # Get number of numeric elements
                    token_size  = hiera_patch_size**2  # Use the hiera patch size we calculated above
                    num_tokens  = total_numel / token_size
                    total_num_tokens += num_tokens

                    # Backward
                    scaler.scale(loss).backward()

                # Increment the grad nosync counter
                grad_nosync_counter += 1

                # Conditional parameter updates when grad sync is required
                if is_grad_sync_required:
                    # ---- Update neural network parameters
                    # Grad clipping
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip) \
                                    if (not uses_dist) or sharding_strategy == ShardingStrategy.NO_SHARD \
                                    else \
                                    model.clip_grad_norm_(grad_clip)

                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()

                    # ---- Report the current iteration
                    # Increment the iteration counter after param update
                    iteration_counter += 1

                    # Obtain the mean total loss
                    if uses_dist:
                        dist.all_reduce(total_loss, op = dist.ReduceOp.AVG)  # Avg across ranks

                    # Obtain the total number of tokens processed
                    if uses_dist:
                        dist.all_reduce(total_num_tokens, op = dist.ReduceOp.SUM)  # Sum across ranks

                    # Wait for all gpus to complete work
                    if device_type == "cuda":
                        torch.cuda.synchronize()

                    # Stop timer
                    t_end = time.monotonic()

                    # Calculate tokens per second
                    t_delta = t_end - t_start
                    tokens_per_sec = total_num_tokens / t_delta

                    # Log the training loop loss after a forward/backward/update
                    if dist_rank == 0:
                        # MFU
                        mfu_per_iteration = estimate_mfu_per_iteration(num_flops_per_token, total_num_tokens, t_delta, peak_flops_per_sec)

                        # Misc
                        current_lrs   = scheduler.get_lr()
                        seg_start_idx = dataset_train.start_idx
                        seg_end_idx   = dataset_train.end_idx

                        # Log
                        log_data = {
                            "rank"               : dist_rank,
                            "logevent"           : "LOSS:TRAIN",
                            "iteration"          : iteration_counter,
                            "segment"            : f"{seg_start_idx}-{seg_end_idx}",
                            "learning_rate"      : ",".join(f"{lr}" for lr in current_lrs),
                            "grad_norm"          : f"{grad_norm:.6f}",
                            "mean_train_loss"    : f"{total_loss:.6f}",
                            "tokens_per_sec"     : f"{tokens_per_sec:.1e}",
                            "mfu_per_iteration"  : f"{mfu_per_iteration:.3f}",
                            "grad_nosync_counter": grad_nosync_counter,
                        }
                        log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
                        logger.info(log_msg)

                    # ---- Monitor training dynamics
                    # Do it before zero-ing gradients
                    if monitors_dynamics and act_monitor is not None:
                        # Monitor preactivation and activation of the nonlinearity
                        for name, act in act_monitor.activations.items():
                            mean_preact, std_preact = act.get('pre')
                            mean_act, std_act       = act.get('pos')
                            log_data = {
                                "rank"        : dist_rank,
                                "iteration"   : iteration_counter,
                                "logevent"    : "DYNAMICS:ACT",
                                "name"        : name,
                                "preact.mean" : mean_preact,
                                "preact.std"  : std_preact,
                                "act.mean"    : mean_act,
                                "act.std"     : std_act,
                            }
                            log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
                            logger.info(log_msg)

                        # Monitor param update - disabled for Hiera (different structure)
                        # current_lr = scheduler.get_lr()[0]  # It's a list
                        # backbone_param_monitor = monitor_param_update_metrics(model.backbone, current_lr)  # Motifs like transformer blocks
                        # Note: Hiera parameter monitoring disabled due to different model structure

                    # ---- Reset for the next iteration
                    # Flush the gradients
                    optimizer.zero_grad(set_to_none = True)

                    # Reset grad accum counter
                    grad_nosync_counter = 0

                    # Reset the loss accumulator
                    total_loss *= 0.0

                    # Reset the token accumulator
                    total_num_tokens *= 0

                    # Reset timer flag
                    starts_timer = True

                    # ---- Update lr every few seg (X segs = one step/iteration)
                    if is_action_due(iteration_counter, scheduler_update_iterations):
                        scheduler.step()
                        if dist_rank == 0:
                            current_lrs = scheduler.get_lr()
                            current_lrs_msg = ",".join(f"{lr}" for lr in current_lrs)
                            logger.info(f"lr is updated to {current_lrs_msg}.")

                    # ---- Eval and checkpointing
                    if is_action_due(iteration_counter, chkpt_saving_iterations):
                        # !!!!!!!!!!!!!!!
                        # !! Data dump !!
                        # !!!!!!!!!!!!!!!
                        data_dump_timestamp = {
                            "uses_dist"       : uses_dist,
                            "dist_rank"       : dist_rank,
                            "dist_world_size" : dist_world_size,
                        }
                        if data_dump_on:
                            data_dump_timestamp.update({
                                "fl_log_prefix"   : fl_log_prefix,
                                "epoch"           : epoch,
                                "seg"             : seg,
                            })

                        if dist_rank == 0:
                            logger.debug('Start evaluation...')

                        # ---- - Eval
                        # ---- -- Train
                        # Get a random subset of the training set
                        train_loss = torch.tensor(float('nan'))
                        num_eval_retry = 0
                        while torch.isnan(train_loss) and (num_eval_retry < max_eval_retry):
                            dataset_eval_train.reset()
                            high_seg_idx = max(dataset_eval_train.total_size - seg_size * dist_world_size, 1)
                            rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,)).item()
                            dataset_eval_train.set_start_idx(rand_start_idx)

                            # Create evaluation dataloader for training set
                            dataloader_eval, sampler_eval = wrap_with_torch_dataloader(
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
                                epoch=rand_start_idx,  # Use rand_start_idx for epoch
                                is_eval=True,
                            )

                            # Set epoch for distributed sampler (handled by utility)
                            if uses_dist and sampler_eval is not None:
                                sampler_eval.set_epoch(rand_start_idx)

                            # Get loss
                            train_loss = estimate_loss(
                                dataloader_eval,
                                model,
                                criterion,
                                autocast_context,
                                max_iter              = max_eval_iter,
                                desc                  = '(training set)',
                                device                = device,
                                dummy_input_shape     = batch_input_shape,
                                mixed_precision_dtype = mixed_precision_dtype,
                                transforms            = transforms,
                                **data_dump_timestamp,
                            )
                            num_eval_retry += 1

                        # Log the train loss
                        if dist_rank == 0:
                            seg_start_idx = dataset_eval_train.start_idx
                            seg_end_idx   = dataset_eval_train.end_idx
                            logger.info(f"LOSS:EVAL - epoch {epoch}, seg {seg_start_idx}-{seg_end_idx}, mean train loss = {train_loss:.8f}")

                        # ---- -- Validation
                        # Get a random subset of the validation set
                        validate_loss = torch.tensor(float('nan'))
                        num_eval_retry = 0
                        while torch.isnan(validate_loss) and (num_eval_retry < max_eval_retry):
                            dataset_eval_val.reset()
                            high_seg_idx = max(dataset_eval_val.total_size - seg_size * dist_world_size, 1)
                            rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,)).item()
                            dataset_eval_val.set_start_idx(rand_start_idx)

                            # Create evaluation dataloader for validation set
                            dataloader_eval, sampler_eval = wrap_with_torch_dataloader(
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
                                epoch=rand_start_idx,  # Use rand_start_idx for epoch
                                is_eval=True,
                            )

                            # Set epoch for distributed sampler (handled by utility)
                            if uses_dist and sampler_eval is not None:
                                sampler_eval.set_epoch(rand_start_idx)

                            validate_loss = estimate_loss(
                                dataloader_eval,
                                model,
                                criterion,
                                autocast_context,
                                max_iter              = max_eval_iter,
                                desc                  = '(validation set)',
                                device                = device,
                                dummy_input_shape     = batch_input_shape,
                                mixed_precision_dtype = mixed_precision_dtype,
                                transforms            = transforms,
                                **data_dump_timestamp,
                            )
                            num_eval_retry += 1

                        # Log the validation loss
                        if dist_rank == 0:
                            seg_start_idx = dataset_eval_val.start_idx
                            seg_end_idx   = dataset_eval_val.end_idx
                            logger.info(f"LOSS:EVAL - epoch {epoch}, seg {seg_start_idx}-{seg_end_idx}, mean validation loss = {validate_loss:.8f}")

                        # ---- - Save checkpoint
                        if validate_loss < loss_min:
                            loss_min = validate_loss

                            # Collect training state
                            iter_state["epoch"]     = epoch
                            iter_state["seg"]       = seg
                            iter_state["start_idx"] = dataset_train.start_idx
                            iter_state["end_idx"]   = dataset_train.end_idx
                            iter_state["loss_min"]  = loss_min

                            dir_chkpt = f"{timestamp}.epoch_{epoch}.end_idx_{dataset_train.end_idx}"
                            if fl_chkpt_prefix is not None: dir_chkpt = f"{fl_chkpt_prefix}.{dir_chkpt}"
                            path_chkpt = os.path.join(dir_root_chkpt, dir_chkpt)
                            checkpointer.save(dist_rank, model, optimizer, scheduler, iter_state, path_chkpt)
                            logger_utils.log_on_all_ranks(logger, f"Saving checkpoint at {path_chkpt}.", "info")

                        # All ranks wait until the end of evaluation by rank 0
                        # [WARNING] Expecting NCCL TIMEOUT ERROR if the evaluation takes too long
                        if uses_dist:
                            dist.barrier()
                        logger.debug('Done evaluation...')

                    # ---- Preemptive checkpointing
                    if preempt_metadata_path is not None and is_action_due(iteration_counter, preempt_chkpt_saving_iterations):
                        # Collect training state
                        iter_state["epoch"]     = epoch
                        iter_state["seg"]       = seg
                        iter_state["start_idx"] = dataset_train.start_idx
                        iter_state["end_idx"]   = dataset_train.end_idx
                        iter_state["loss_min"]  = loss_min

                        dir_chkpt = f"{timestamp}.preempt"
                        if fl_chkpt_prefix is not None: dir_chkpt = f"{fl_chkpt_prefix}.{dir_chkpt}"
                        path_chkpt = os.path.join(dir_root_chkpt, dir_chkpt)
                        checkpointer.save(dist_rank, model, optimizer, scheduler, iter_state, path_chkpt)
                        logger_utils.log_on_all_ranks(logger, f"Saving preemptive checkpoint (epoch {epoch}, end_idx {dataset_train.end_idx}) at {path_chkpt}.", "info")

                        if dist_rank == 0:
                            with open(preempt_metadata_path, "w") as f:
                                f.write(path_chkpt)
                            logger.info(f"Saving preemptive metadata (epoch {epoch}, end_idx {dataset_train.end_idx}) at {preempt_metadata_path}.")

            # [PERFORMANCE]
            if dist_local_rank == 0:
                memmax.update()

            # [PERFORMANCE]
            if dist_local_rank == 0:
                memmax.stop()

        # Reset last_seg
        last_seg = -1

        # Reset the from_resume flag
        from_resume = False

except KeyboardInterrupt:
    logger_utils.log_on_all_ranks(logger, "Training was interrupted!", "error")
except Exception as e:
    tb = traceback.format_exc()
    logger_utils.log_on_all_ranks(logger, f"Error occurred: {e}\nTraceback: {tb}", "error")
finally:
    # Clean up hooks
    if monitors_dynamics and act_monitor is not None:
        act_monitor.remove_hooks()

    # Ensure that the process group is always destroyed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
