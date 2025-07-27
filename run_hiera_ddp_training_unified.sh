#!/bin/bash

RUNS_NSYS=0
NUM_MPI_TASKS=10  # Use all 10 GPUs

# [NEW OPTIONS]
TARGET_FACILITY="s3df"        # Which facility to target (s3df, nersc, summit)
GENERATE_ONLY=false           # Set true to only generate scripts
AUTO_SUBMIT=false            # Set true to submit job instead of run locally

# [JOB CONFIG]
JOB=hiera-ddp-training
WALLTIME=48:00:00
PARTITION=ada
QOS=""

# [TRAIN]
BATCH_SIZE=40
H_PAD=1920
W_PAD=1920
HV=512  # Polar crop size
WV=512
NUM_WORKERS=2
USES_PAD=true
USES_POLAR_CENTER_CROP=true
USES_BATCH_SAMPLER=false
USES_RANDOM_PATCH=false
USES_RANDOM_ROTATE=false
USES_RANDOM_SHIFT=false
PATH_CHKPT_PREV=null
PREEMPT_CHKPT_SAVING_STEPS=20  # Save every 20 iterations
CHKPT_SAVING_STEPS=100
GRAD_ACCUM_STEPS=1
WARMUP=1000
SHARDING_STAGE="zero0"  # DDP mode
OFFLOAD_TO_CPU=false

# [MODEL - HIERA HUGE]
EMBED_DIM=192
NUM_HEADS=3
STAGES="[2, 6, 48, 4]"  # Hiera-Huge configuration
DIM_MUL=2.0
HEAD_MUL=2.0
DECODER_EMBED_DIM=768
DECODER_DEPTH=12
DECODER_NUM_HEADS=24

# [DATASET] - Using working zarr dataset
PATH_TRAIN="pretrain/train.csv"
PATH_EVAL="pretrain/eval.csv"
ENABLE_SHUFFLING_TRAIN=true
ENABLE_SHUFFLING_EVAL=false  # Keep eval deterministic
SHUFFLE_SEED_BASE=42
RESHUFFLE_FREQUENCY=10  # Reshuffle every 10 steps for testing

# [LOSS]
FOCAL_ALPHA="[0.25, 0.75]"
FOCAL_GAMMA=2.0

PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
PREEMPT_METADATA_PATH="$PREEMPT_ROOT/$JOB"

echo "🚀 Unified Hiera DDP Training Script"
echo "==========================================="
echo "Target Facility: $TARGET_FACILITY"
echo "Generate Only: $GENERATE_ONLY"
echo "Auto Submit: $AUTO_SUBMIT"
echo "Job: $JOB"
echo ""

# Generate job scripts for all facilities using unified launcher
echo "📦 Generating job scripts with unified launcher..."
python launch_unified.py \
job=$JOB \
auto_submit=$AUTO_SUBMIT \
target_facility=$TARGET_FACILITY \
resource_configs=hiera \
train_config=hiera \
train_config.checkpoint.prefix=$JOB \
train_config.checkpoint.path_chkpt_prev=$PATH_CHKPT_PREV \
train_config.checkpoint.state_dict_type=sharded \
train_config.checkpoint.preempt_metadata_path=$PREEMPT_METADATA_PATH \
train_config.checkpoint.preempt_chkpt_saving_steps=$PREEMPT_CHKPT_SAVING_STEPS \
train_config.checkpoint.chkpt_saving_steps=$CHKPT_SAVING_STEPS \
train_config.checkpoint.offload_to_cpu=$OFFLOAD_TO_CPU \
"train_config.dataset.path_train=$PATH_TRAIN" \
"train_config.dataset.path_eval=$PATH_EVAL" \
train_config.dataset.enable_shuffling_train=$ENABLE_SHUFFLING_TRAIN \
train_config.dataset.enable_shuffling_eval=$ENABLE_SHUFFLING_EVAL \
train_config.dataset.shuffle_seed_base=$SHUFFLE_SEED_BASE \
train_config.dataset.reshuffle_frequency=$RESHUFFLE_FREQUENCY \
train_config.dataset.num_workers=$NUM_WORKERS \
train_config.dataset.prefetch_factor=4 \
train_config.dataset.pin_memory=true \
train_config.dataset.batch_size=$BATCH_SIZE \
train_config.dataset.transforms.set.pad=$USES_PAD \
train_config.dataset.transforms.set.polar_center_crop=$USES_POLAR_CENTER_CROP \
train_config.dataset.transforms.set.batch_sampler=$USES_BATCH_SAMPLER \
train_config.dataset.transforms.set.random_patch=$USES_RANDOM_PATCH \
train_config.dataset.transforms.set.random_rotate=$USES_RANDOM_ROTATE \
train_config.dataset.transforms.set.random_shift=$USES_RANDOM_SHIFT \
train_config.dataset.transforms.H_pad=$H_PAD \
train_config.dataset.transforms.W_pad=$W_PAD \
train_config.dataset.transforms.Hv=$HV \
train_config.dataset.transforms.Wv=$WV \
train_config.model.hiera.embed_dim=$EMBED_DIM \
train_config.model.hiera.num_heads=$NUM_HEADS \
"train_config.model.hiera.stages=$STAGES" \
train_config.model.hiera.dim_mul=$DIM_MUL \
train_config.model.hiera.head_mul=$HEAD_MUL \
train_config.model.hiera.decoder_embed_dim=$DECODER_EMBED_DIM \
train_config.model.hiera.decoder_depth=$DECODER_DEPTH \
train_config.model.hiera.decoder_num_heads=$DECODER_NUM_HEADS \
train_config.loss.grad_accum_steps=$GRAD_ACCUM_STEPS \
"train_config.loss.focal.alpha=$FOCAL_ALPHA" \
train_config.loss.focal.gamma=$FOCAL_GAMMA \
train_config.optim.lr=0.0001 \
train_config.optim.fused=false \
train_config.misc.monitors_dynamics=true \
train_config.misc.compiles_model=false \
train_config.misc.max_eval_iter=10 \
"train_config.misc.sharding_stage=$SHARDING_STAGE" \
train_config.misc.data_dump_on=false \
train_config.lr_scheduler.warmup_steps=$WARMUP \
train_config.lr_scheduler.total_steps=10000000 \
train_config.lr_scheduler.scheduler_update_steps=1 \
train_config.logging.prefix=$JOB \
train_config.dist.dtype=bfloat16

launcher_exit_code=$?
if [ $launcher_exit_code -ne 0 ]; then
    echo "❌ Failed to generate job scripts (exit code: $launcher_exit_code)"
    exit $launcher_exit_code
fi

echo ""
echo "✅ Job scripts generated!"

# Show generated scripts
echo "📄 Generated job scripts:"
if [ -f "experiments/jobs/$JOB.s3df.sbatch" ]; then
    echo "   🖥️  SLAC S3DF: experiments/jobs/$JOB.s3df.sbatch"
fi
if [ -f "experiments/jobs/$JOB.nersc.sbatch" ]; then
    echo "   🌊 NERSC: experiments/jobs/$JOB.nersc.sbatch"
fi
if [ -f "experiments/jobs/$JOB.summit.bsub" ]; then
    echo "   🏔️  ORNL Summit: experiments/jobs/$JOB.summit.bsub"
fi

# If AUTO_SUBMIT is true, the launcher already submitted the job, so we're done
if [ "$AUTO_SUBMIT" = true ]; then
    echo ""
    echo "🚀 Job submitted automatically via unified launcher!"
    exit 0
fi

# If GENERATE_ONLY is true, stop here
if [ "$GENERATE_ONLY" = true ]; then
    echo ""
    echo "🛑 Generate-only mode: Job scripts created but not executed."
    echo "💡 To submit manually:"
    if [ -f "experiments/jobs/$JOB.s3df.sbatch" ]; then
        echo "   sbatch experiments/jobs/$JOB.s3df.sbatch"
    fi
    if [ -f "experiments/jobs/$JOB.nersc.sbatch" ]; then
        echo "   sbatch experiments/jobs/$JOB.nersc.sbatch"
    fi
    if [ -f "experiments/jobs/$JOB.summit.bsub" ]; then
        echo "   bsub experiments/jobs/$JOB.summit.bsub"
    fi
    exit 0
fi

# Otherwise, run training directly (original behavior)
echo ""
echo "🏃 Running training directly (local execution)..."

base_command="torchrun --nproc_per_node=$NUM_MPI_TASKS --nnodes=1 --node_rank=0 train_hiera_seg.py experiments/yaml/$JOB.yaml"
## base_command="mpirun -n $NUM_MPI_TASKS --map-by ppr:${NUM_MPI_TASKS}:node --bind-to none python train_hiera_seg.py experiments/yaml/$JOB.yaml"
final_command="OMP_NUM_THREADS=1 "

if [ $RUNS_NSYS -eq 1 ]; then
    final_command+="nsys profile -w true -t cuda --wait primary -o /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/nsight_reports/$JOB.profile -f true --cudabacktrace=true -x true "
fi
final_command+="$base_command"

echo "Launching training with command: $final_command"
eval $final_command