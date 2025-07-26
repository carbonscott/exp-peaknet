#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hiera Segmentation Training with Hugging Face Accelerate

This script replaces the manual FSDP/distributed training setup with
Hugging Face Accelerate for simplified distributed training.

Key improvements over train_hiera_seg.py:
- Replaces ~300 lines of manual FSDP setup with Accelerator()
- Simplified checkpointing with accelerator.save_state()/load_state()
- Built-in mixed precision and gradient accumulation
- Automatic device placement and gradient scaling
- Uses estimate_loss_accelerate for robust distributed evaluation
- Works with single GPU, DDP, FSDP automatically based on accelerate config
"""

# -- Basic imports
import os
import argparse
import inspect
import logging
import time
from functools import partial
from omegaconf import OmegaConf

# -- Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed

# -- peaknet specific imports
# --- Dataset
from peaknet.datasets.data import (
    PeakNetDatasetConfig,
    PeakNetDataset,
)
from peaknet.datasets.checkpoints import CheckpointManager
from peaknet.tensor_transforms import (
    Pad,
    PolarCenterCrop,
    MergeBatchPatchDims,
)

# --- Model (HieraBlock, MaskedAutoencoderHiera needed for FSDP compatibility)
from peaknet.modeling.hiera_segmentation import HieraSegmentation
from peaknet.modeling.hiera import Hiera, HieraBlock
from peaknet.modeling.hiera_mae import MaskedAutoencoderHiera

# --- Loss
from peaknet.criterion import CategoricalFocalLoss

# --- Others
from peaknet.utils.misc import is_action_due
from peaknet.utils.data import wrap_with_torch_dataloader
from peaknet.lr_scheduler import CosineLRScheduler
from peaknet.utils.signal import register_handlers
from peaknet.utils.eval import estimate_loss_accelerate

# -- Torch specific imports
import torch
import torch.optim as optim

# -- Debug
torch.autograd.set_detect_anomaly(False)

# ----------------------------------------------------------------------- #
#  STRESS TESTING HELPERS
# ----------------------------------------------------------------------- #
def log_gpu_memory_usage(accelerator, stage=""):
    """Log GPU memory usage per rank for FSDP verification."""
    if torch.cuda.is_available():
        rank = accelerator.process_index
        device = accelerator.device
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        
        if accelerator.is_main_process:
            logger.info(f"=== GPU MEMORY USAGE {stage} ===")
        
        # Log from all ranks but with rank identification
        logger.info(f"[RANK {rank}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
        
        # Barrier to organize output
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()

def log_step_details(accelerator, step, loss, lr=None, extra_info=""):
    """Enhanced logging for stress testing with exact loss values."""
    if accelerator.is_main_process:
        msg = f"[STRESS TEST] Step {step}: loss={loss:.8f}"
        if lr is not None:
            msg += f", lr={lr:.2e}"
        if extra_info:
            msg += f" ({extra_info})"
        logger.info(msg)

# -- Get the logger
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  COMMAND LINE INTERFACE
# ----------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Hiera segmentation training with Accelerate")
parser.add_argument("yaml_file", help="Path to the YAML config file")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                   help="Path to checkpoint folder to resume from")
args = parser.parse_args()

# ----------------------------------------------------------------------- #
#  CONFIGURATION
# ----------------------------------------------------------------------- #
# Load CONFIG from YAML using OmegaConf
fl_yaml = args.yaml_file
config = OmegaConf.load(fl_yaml)

# -- Checkpoint
dir_root_chkpt = config.checkpoint.directory
fl_chkpt_prefix = config.checkpoint.prefix
chkpt_saving_steps = config.checkpoint.chkpt_saving_steps
preempt_metadata_path = config.checkpoint.get("preempt_metadata_path", os.environ.get('PREEMPT_METADATA_PATH', None))
preempt_chkpt_saving_steps = config.checkpoint.preempt_chkpt_saving_steps

# -- Dataset (keep existing config)
path_dataset_train = config.dataset.path_train
path_dataset_eval = config.dataset.path_eval
drop_last_in_sampler = config.dataset.drop_last_in_sampler
drop_last_in_loader = config.dataset.drop_last_in_loader
batch_size = config.dataset.batch_size
seg_size = config.dataset.seg_size
num_workers = config.dataset.num_workers
pin_memory = config.dataset.pin_memory
prefetch_factor = config.dataset.prefetch_factor
cache_size = config.dataset.cache_size

# Transform parameters (keep existing)
H_pad = config.dataset.transforms.H_pad
W_pad = config.dataset.transforms.W_pad
Hv = config.dataset.transforms.Hv
Wv = config.dataset.transforms.Wv
sigma = config.dataset.transforms.sigma
num_crop = config.dataset.transforms.num_crop
uses_pad = config.dataset.transforms.set.pad
uses_polar_center_crop = config.dataset.transforms.set.polar_center_crop

# -- Model parameters (keep existing)
num_classes = config.model.hiera.get("num_classes", 2)
decoder_embed_dim = config.model.hiera.get("decoder_embed_dim", 512)
decoder_depth = config.model.hiera.get("decoder_depth", 8)
decoder_num_heads = config.model.hiera.get("decoder_num_heads", 16)
input_size = tuple(config.model.hiera.get("input_size", [224, 224]))
in_chans = config.model.hiera.get("in_chans", 1)
embed_dim = config.model.hiera.get("embed_dim", 96)
num_heads = config.model.hiera.get("num_heads", 1)
stages = tuple(config.model.hiera.get("stages", [2, 3, 16, 3]))
q_pool = config.model.hiera.get("q_pool", 3)
q_stride = tuple(config.model.hiera.get("q_stride", [2, 2]))
patch_stride = tuple(config.model.hiera.get("patch_stride", [4, 4]))

# -- Loss
grad_accum_steps = max(int(config.loss.grad_accum_steps), 1)
focal_alpha = config.loss.focal.alpha
focal_gamma = config.loss.focal.gamma

# -- Optimizer
lr = float(config.optim.lr)
weight_decay = float(config.optim.weight_decay)
adam_beta1 = float(config.optim.beta1)
adam_beta2 = float(config.optim.beta2)
grad_clip = float(config.optim.grad_clip)

# -- Scheduler
warmup_steps = config.lr_scheduler.warmup_steps
total_steps = config.lr_scheduler.total_steps
min_lr = float(config.lr_scheduler.min_lr)
scheduler_update_steps = config.lr_scheduler.scheduler_update_steps

# -- Logging
drc_log = config.logging.directory
fl_log_prefix = config.logging.prefix
log_level = config.logging.level

# -- Misc
max_eval_iter = config.misc.max_eval_iter
compiles_model = config.misc.compiles_model

# ----------------------------------------------------------------------- #
#  ACCELERATE SETUP
# ----------------------------------------------------------------------- #
# Register signal handlers for graceful shutdown
register_handlers()

# Initialize Accelerator - this replaces ALL manual distributed setup!
# No more dist_rank, dist_world_size, FSDP wrapping, mixed precision setup, etc.
# All configuration now handled by accelerate config file
accelerator = Accelerator(
    gradient_accumulation_steps=grad_accum_steps,
    project_dir=drc_log
)

# Set up logging (only on main process)
if accelerator.is_main_process:
    os.makedirs(drc_log, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(drc_log, f"{fl_log_prefix}_accelerate.log")),
            logging.StreamHandler()
        ]
    )
    logger.info("=" * 60)
    logger.info("TRAINING WITH HUGGING FACE ACCELERATE")
    logger.info(f"Device: {accelerator.device}")
    logger.info(f"Distributed type: {accelerator.distributed_type}")
    logger.info(f"Num processes: {accelerator.num_processes}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision} (config-driven)")
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logger.info("=" * 60)

# ----------------------------------------------------------------------- #
#  DATASET SETUP (keep existing logic)
# ----------------------------------------------------------------------- #
logger.debug('Setting up dataset...')

# Seed setup - use accelerator's process index instead of manual dist_rank
base_seed = 0
world_seed = base_seed + accelerator.process_index
set_seed(world_seed)  # Use accelerate's set_seed

# Set up transformations (keep existing)
class NoTransform:
    def __call__(self, x, **kwargs):
        return x

merges_batch_patch_dims = uses_polar_center_crop
pre_transforms = (
    Pad(H_pad, W_pad) if uses_pad else NoTransform(),
)

transforms = (
    PolarCenterCrop(
        Hv=Hv,
        Wv=Wv,
        sigma=sigma,
        num_crop=num_crop,
    ) if uses_polar_center_crop else NoTransform(),
    MergeBatchPatchDims() if merges_batch_patch_dims else NoTransform(),
)

# Set up datasets (modernized with PeakNetDataset)
dataset_train_config = PeakNetDatasetConfig(
    path_csv=path_dataset_train,
    transforms=pre_transforms,
    buffer_size=cache_size,
    dist_rank=accelerator.process_index,  # Use accelerator instead of manual dist_rank
    dist_world_size=accelerator.num_processes,  # Use accelerator instead of manual dist_world_size
    device=str(accelerator.device),
    dtype=None,
    uses_norm=True,
    scales_variance=True,
    perfs_runtime=False,
)
dataset_train = PeakNetDataset(dataset_train_config)

# Set up eval datasets
dataset_eval_train = PeakNetDataset(dataset_train_config)

dataset_eval_val_config = PeakNetDatasetConfig(
    path_csv=path_dataset_eval,
    transforms=pre_transforms,
    buffer_size=cache_size,
    dist_rank=accelerator.process_index,
    dist_world_size=accelerator.num_processes,
    device=str(accelerator.device),
    dtype=None,
    uses_norm=True,
    scales_variance=True,
    perfs_runtime=False,
)
dataset_eval_val = PeakNetDataset(dataset_eval_val_config)

# Set up checkpoint manager for dataset state
checkpoint_manager = CheckpointManager(accelerator.process_index, accelerator.num_processes)

# ----------------------------------------------------------------------- #
#  CHECKPOINT UTILITIES
# ----------------------------------------------------------------------- #
def extract_timestamp_from_checkpoint(checkpoint_path):
    """Extract timestamp from checkpoint filename.

    Examples:
    - "hiera-test_2024_0115_1430_step_12345" → "2024_0115_1430"
    - "hiera-test_2024_0115_1430_best_step_10000" → "2024_0115_1430"
    - "hiera-test_2024_0115_1430_step_12345.preempt" → "2024_0115_1430"
    """
    basename = os.path.basename(checkpoint_path)
    if "_step_" in basename:
        prefix_and_timestamp = basename.split("_step_")[0]
        parts = prefix_and_timestamp.split("_")
        if len(parts) >= 3:
            # Last 3 parts form timestamp: YYYY_MMDD_HHMM
            return "_".join(parts[-3:])
    return None

def parse_step_from_checkpoint(checkpoint_path):
    """Parse step number from any checkpoint type.

    Examples:
    - "hiera-test_2024_0115_1430_step_12345" → 12345
    - "hiera-test_2024_0115_1430_best_step_10000" → 10000
    - "hiera-test_2024_0115_1430_step_12345.preempt" → 12345
    """
    basename = os.path.basename(checkpoint_path)
    if "step_" in basename:
        return int(basename.split("step_")[-1].split(".")[0])
    return 0

def broadcast_timestamp_to_all_ranks(timestamp):
    """Broadcast timestamp from main process to all ranks for consistency."""
    if accelerator.num_processes > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            if accelerator.is_main_process:
                # Pack timestamp as integer: 202401151430
                timestamp_int = int(timestamp.replace("_", ""))
            else:
                timestamp_int = 0

            timestamp_tensor = torch.tensor([timestamp_int], dtype=torch.long, device=accelerator.device)
            dist.broadcast(timestamp_tensor, src=0)

            # Unpack back to format: 202401151430 → "2024_0115_1430"
            timestamp_str = str(timestamp_tensor.item()).zfill(12)  # Pad to 12 digits
            return f"{timestamp_str[:4]}_{timestamp_str[4:8]}_{timestamp_str[8:12]}"

    return timestamp

def create_infinite_dataloader(dataset, base_seed, drop_last_in_sampler, drop_last_in_loader, 
                              uses_dist, batch_size, num_workers, pin_memory, prefetch_factor):
    """Create an infinite cycling dataloader that repeats the dataset indefinitely."""
    from itertools import cycle

    # Create base dataloader
    dataloader, sampler = wrap_with_torch_dataloader(
        dataset=dataset,
        base_seed=base_seed,
        drop_last_in_sampler=drop_last_in_sampler,
        drop_last_in_loader=drop_last_in_loader,
        uses_dist=uses_dist,
        batch_size=batch_size,
        num_workers=num_workers,
        custom_collate=None,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        epoch=0,  # Fixed epoch for cycling
        is_eval=False,
    )

    # Create infinite cycling iterator
    return cycle(dataloader), sampler

# ----------------------------------------------------------------------- #
#  MODEL SETUP
# ----------------------------------------------------------------------- #
logger.debug('Setting up model...')

# Create model (keep existing logic)
model = HieraSegmentation(
    num_classes=num_classes,
    decoder_embed_dim=decoder_embed_dim,
    decoder_depth=decoder_depth,
    decoder_num_heads=decoder_num_heads,
    input_size=input_size,
    in_chans=in_chans,
    embed_dim=embed_dim,
    num_heads=num_heads,
    stages=stages,
    q_pool=q_pool,
    q_stride=q_stride,
    patch_stride=patch_stride,
)

# Initialize weights
if hasattr(model, 'init_weights') and callable(model.init_weights):
    model.init_weights()

if accelerator.is_main_process:
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model has {total_params:.1f}M parameters")

# Compile model if requested
if compiles_model:
    logger.debug("Compiling model...")
    model = torch.compile(model)

# ----------------------------------------------------------------------- #
#  OPTIMIZER AND SCHEDULER
# ----------------------------------------------------------------------- #
logger.debug('Setting up optimizer and scheduler...')

# Create optimizer (keep existing logic)
optim_arg_dict = dict(
    lr=lr,
    weight_decay=weight_decay,
    betas=(adam_beta1, adam_beta2),
)
if 'fused' in inspect.signature(optim.AdamW).parameters:
    optim_arg_dict['fused'] = True

optimizer = optim.AdamW(model.parameters(), **optim_arg_dict)

# Create scheduler (keep existing)
scheduler = CosineLRScheduler(
    optimizer=optimizer,
    warmup_iterations=warmup_steps,
    total_iterations=total_steps,
    min_lr=min_lr
)

# ----------------------------------------------------------------------- #
#  CRITERION
# ----------------------------------------------------------------------- #
criterion = CategoricalFocalLoss(
    alpha=focal_alpha,
    gamma=focal_gamma,
    num_classes=num_classes,
)

# ----------------------------------------------------------------------- #
#  PREPARE WITH ACCELERATE
# ----------------------------------------------------------------------- #
model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
logger.debug('Model, optimizer, and scheduler prepared with Accelerate')

# Log GPU memory usage after model preparation (for FSDP sharding verification)
log_gpu_memory_usage(accelerator, "AFTER MODEL PREPARATION")

# ----------------------------------------------------------------------- #
#  TIMESTAMP GENERATION AND RESUMPTION
# ----------------------------------------------------------------------- #
starting_step = 0
loss_min = float('inf')
run_timestamp = None

# Determine if we're resuming and extract timestamp + step
if args.resume_from_checkpoint:
    # Explicit resumption from any checkpoint type
    starting_step = parse_step_from_checkpoint(args.resume_from_checkpoint) + 1
    run_timestamp = extract_timestamp_from_checkpoint(args.resume_from_checkpoint)
    if not run_timestamp:
        # Fallback to current time if timestamp can't be parsed
        run_timestamp = time.strftime("%Y_%m%d_%H%M")
        if accelerator.is_main_process:
            logger.warning(f"Could not extract timestamp from {args.resume_from_checkpoint}, using current time")
elif preempt_metadata_path and os.path.exists(preempt_metadata_path):
    # Auto-resume from latest preemptive checkpoint
    try:
        with open(preempt_metadata_path, 'r') as f:
            preempt_checkpoint_path = f.read().strip()
        if os.path.exists(preempt_checkpoint_path):
            args.resume_from_checkpoint = preempt_checkpoint_path  # Set for later use
            starting_step = parse_step_from_checkpoint(preempt_checkpoint_path) + 1
            run_timestamp = extract_timestamp_from_checkpoint(preempt_checkpoint_path)
            if accelerator.is_main_process:
                logger.info(f"Auto-resuming from preemptive checkpoint: {preempt_checkpoint_path}")
    except Exception as e:
        if accelerator.is_main_process:
            logger.warning(f"Failed to read preemptive metadata from {preempt_metadata_path}: {e}")

# Generate new timestamp if this is a fresh run
if run_timestamp is None:
    if accelerator.is_main_process:
        run_timestamp = time.strftime("%Y_%m%d_%H%M")
    else:
        run_timestamp = None
    # Broadcast timestamp to all ranks for consistency
    run_timestamp = broadcast_timestamp_to_all_ranks(run_timestamp)

if accelerator.is_main_process:
    logger.info(f"Run timestamp: {run_timestamp}")
    logger.info(f"Starting from step: {starting_step}")

# Create a simple state object for dataset checkpointing
class DatasetState:
    def __init__(self):
        self.dataset_checkpoint = None
        self.loss_min = float('inf')

    def state_dict(self):
        if hasattr(dataset_train, 'get_checkpoint_state'):
            dataset_local_state = dataset_train.get_checkpoint_state()
            self.dataset_checkpoint = checkpoint_manager.aggregate_global_progress(dataset_local_state)
        return {
            'dataset_checkpoint': self.dataset_checkpoint,
            'loss_min': self.loss_min
        }

    def load_state_dict(self, state_dict):
        self.dataset_checkpoint = state_dict.get('dataset_checkpoint')
        self.loss_min = state_dict.get('loss_min', float('inf'))

dataset_state = DatasetState()

# Register dataset state for automatic checkpointing with Accelerate
accelerator.register_for_checkpointing(dataset_state)

# Load checkpoint if resuming (step parsing already done above)
if args.resume_from_checkpoint:
    accelerator.load_state(args.resume_from_checkpoint)

    # Restore dataset checkpoint state if available
    if dataset_state.dataset_checkpoint is not None:
        # Use CheckpointManager to coordinate resumption (handles variable GPU counts)
        resumption_config = checkpoint_manager.coordinate_resumption(
            {"dataset_state": dataset_state.dataset_checkpoint}, 
            accelerator.num_processes, 
            accelerator.process_index
        )

        if resumption_config.get("resumption_successful", False):
            if accelerator.is_main_process:
                logger.info(f"Dataset resumption: {resumption_config.get('samples_for_this_rank', 0)} samples assigned to rank {accelerator.process_index}")

    loss_min = dataset_state.loss_min

    if accelerator.is_main_process:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        logger.info(f"Resumed loss_min: {loss_min}")

# ----------------------------------------------------------------------- #
#  TRAINING LOOP
# ----------------------------------------------------------------------- #
logger.debug('Starting step-based training loop...')

# Initialize step counter from resumption
step_counter = starting_step

# Create infinite cycling dataloader
infinite_dataloader, sampler = create_infinite_dataloader(
    dataset=dataset_train,
    base_seed=base_seed,
    drop_last_in_sampler=drop_last_in_sampler,
    drop_last_in_loader=drop_last_in_loader,
    uses_dist=accelerator.num_processes > 1,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
)

# Prepare dataloader with accelerator
infinite_dataloader = accelerator.prepare(infinite_dataloader)

if accelerator.is_main_process:
    logger.info(f"Training for {total_steps} total steps (starting from step {starting_step})")

try:
    model.train()

    # Step-based training loop
    for batch_data in infinite_dataloader:
        if step_counter >= total_steps:
            break

        if batch_data is None:
            continue

        # Process batch data (keep existing logic)
        batch_data = torch.cat(batch_data, dim=0)  # (2*B, C, H, W)

        # Apply transforms
        if transforms is not None:
            for trans in transforms:
                batch_data = trans(batch_data)

        # Split input and target
        current_batch_size = batch_data.size(0) // 2
        batch_input = batch_data[:current_batch_size]
        batch_target = batch_data[current_batch_size:]

        # Binarize labels
        if transforms is not None:
            batch_target = batch_target > 0.5

        # Forward and backward pass with accelerator
        with accelerator.accumulate(model):
            outputs = model(batch_input)
            loss = criterion(outputs, batch_target)
            loss = loss.mean()

            # Use accelerator.backward() instead of loss.backward()
            accelerator.backward(loss)

            # Gradient clipping
            if grad_clip > 0.0:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

        # Track dataset progress for checkpointing
        dataset_train.advance_local_progress(current_batch_size)

        step_counter += 1

        # Update scheduler
        if is_action_due(step_counter, scheduler_update_steps):
            scheduler.step()
            if accelerator.is_main_process:
                current_lrs = scheduler.get_lr()
                current_lrs_msg = ",".join(f"{lr}" for lr in current_lrs)
                logger.info(f"lr updated to {current_lrs_msg}")

        # Logging (only on main process)
        if accelerator.is_main_process and step_counter % 10 == 0:
            current_lrs = scheduler.get_lr()
            logger.info(f"Step {step_counter}: loss={loss:.6f}, lr={current_lrs[0]:.2e}")
        
        # Enhanced logging for stress testing (every step near checkpoints)
        if step_counter % chkpt_saving_steps == 0 or step_counter % chkpt_saving_steps == 1:
            current_lrs = scheduler.get_lr()
            log_step_details(accelerator, step_counter, loss, current_lrs[0], "checkpoint boundary")

        # Checkpointing
        if is_action_due(step_counter, chkpt_saving_steps):
            # Evaluation
            model.eval()

            with torch.no_grad():
                # Evaluate on training set using estimate_loss_accelerate
                eval_dataloader_train, _ = wrap_with_torch_dataloader(
                    dataset=dataset_eval_train,
                    base_seed=base_seed,
                    drop_last_in_sampler=drop_last_in_sampler,
                    drop_last_in_loader=drop_last_in_loader,
                    uses_dist=accelerator.num_processes > 1,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    custom_collate=None,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                    epoch=0,  # Fixed epoch for eval
                    is_eval=True,
                )
                eval_dataloader_train = accelerator.prepare(eval_dataloader_train)

                train_eval_loss = estimate_loss_accelerate(
                    eval_dataloader_train,
                    model,
                    criterion,
                    accelerator,
                    max_iter=max_eval_iter,
                    desc='(training set)',
                    transforms=transforms
                )

                # Evaluate on validation set using estimate_loss_accelerate
                eval_dataloader_val, _ = wrap_with_torch_dataloader(
                    dataset=dataset_eval_val,
                    base_seed=base_seed,
                    drop_last_in_sampler=drop_last_in_sampler,
                    drop_last_in_loader=drop_last_in_loader,
                    uses_dist=accelerator.num_processes > 1,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    custom_collate=None,
                    pin_memory=pin_memory,
                    prefetch_factor=prefetch_factor,
                    epoch=0,  # Fixed epoch for eval
                    is_eval=True,
                )
                eval_dataloader_val = accelerator.prepare(eval_dataloader_val)

                val_eval_loss = estimate_loss_accelerate(
                    eval_dataloader_val,
                    model,
                    criterion,
                    accelerator,
                    max_iter=max_eval_iter,
                    desc='(validation set)',
                    transforms=transforms
                )

            # Process evaluation results (estimate_loss_accelerate returns properly averaged losses)
            # Use validation loss for model selection if available, otherwise use training loss
            if not torch.isnan(val_eval_loss) and not torch.isinf(val_eval_loss):
                eval_loss = val_eval_loss
            else:
                eval_loss = train_eval_loss if not torch.isnan(train_eval_loss) and not torch.isinf(train_eval_loss) else float('inf')

            if accelerator.is_main_process:
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
                dataset_state.loss_min = eval_loss
                best_output_dir = f"{fl_chkpt_prefix}_{run_timestamp}_best_step_{step_counter}"
                best_output_path = os.path.join(dir_root_chkpt, best_output_dir)
                accelerator.save_state(best_output_path)

                if accelerator.is_main_process:
                    logger.info(f"New best model saved: {best_output_dir} (val_loss={eval_loss:.6f})")

            # Regular checkpoint with unified naming
            regular_output_dir = f"{fl_chkpt_prefix}_{run_timestamp}_step_{step_counter}"
            regular_output_path = os.path.join(dir_root_chkpt, regular_output_dir)
            accelerator.save_state(regular_output_path)

            if accelerator.is_main_process:
                logger.info(f"Regular checkpoint saved: {regular_output_dir}")

            model.train()  # Back to training mode

        # Preemptive checkpointing (for HPC time limits)
        if preempt_metadata_path and is_action_due(step_counter, preempt_chkpt_saving_steps):
            preempt_output_dir = f"{fl_chkpt_prefix}_{run_timestamp}_step_{step_counter}.preempt"
            preempt_output_path = os.path.join(dir_root_chkpt, preempt_output_dir)
            accelerator.save_state(preempt_output_path)

            # Write metadata file with checkpoint path
            if accelerator.is_main_process:
                with open(preempt_metadata_path, "w") as f:
                    f.write(preempt_output_path)
                logger.info(f"Preemptive checkpoint saved: {preempt_output_dir}")
                logger.info(f"Preemptive metadata written to: {preempt_metadata_path}")

    # Training completed
    if accelerator.is_main_process:
        logger.info(f"Training completed after {step_counter} steps")

except KeyboardInterrupt:
    if accelerator.is_main_process:
        logger.info("Training interrupted by user")
except Exception as e:
    if accelerator.is_main_process:
        logger.error(f"Training failed with error: {e}")
        logger.error(traceback.format_exc())
    raise
finally:
    # Clean up
    accelerator.end_training()
    if accelerator.is_main_process:
        logger.info("Training completed")
