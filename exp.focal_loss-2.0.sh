#!/bin/bash

RUNS_NSYS=0
NUM_MPI_TASKS=4

JOB=focal_loss-2.0
FOCAL_ALPHA="[0.5, 0.5]"
FOCAL_GAMMA=2
BATCH_SIZE=6
H_PAD=1920
W_PAD=1920
NUM_WORKERS=2
USES_PAD=true
USES_POLAR_CENTER_CROP=false
USES_BATCH_SAMPLER=false
USES_RANDOM_PATCH=true
USES_RANDOM_ROTATE=true
USES_RANDOM_SHIFT=true
PATH_CHKPT_PREV=null
PREEEMPT_CHKPT_SAVING_ITERATIONS=20

PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
PREEMPT_METADATA_PATH="$PREEMPT_ROOT/$JOB"

SEG_SIZE=$((BATCH_SIZE * 60))

python launch_job.slurm.py \
job=$JOB \
auto_submit=false \
sbatch_config.trainer=train.fsdp.py \
train_config.checkpoint.prefix=$JOB \
train_config.checkpoint.path_chkpt_prev=$PATH_CHKPT_PREV \
train_config.checkpoint.state_dict_type=full \
train_config.checkpoint.preempt_metadata_path=$PREEMPT_METADATA_PATH \
train_config.checkpoint.preempt_chkpt_saving_iterations=$PREEEMPT_CHKPT_SAVING_ITERATIONS \
train_config.checkpoint.chkpt_saving_iterations=20 \
train_config.dataset.path_train=train.csv \
train_config.dataset.path_eval=eval.csv \
train_config.dataset.num_workers=$NUM_WORKERS \
train_config.dataset.prefetch_factor=10 \
train_config.dataset.pin_memory=true \
train_config.dataset.seg_size=$SEG_SIZE \
train_config.dataset.batch_size=$BATCH_SIZE \
train_config.dataset.transforms.set.pad=$USES_PAD \
train_config.dataset.transforms.set.polar_center_crop=$USES_POLAR_CENTER_CROP \
train_config.dataset.transforms.set.batch_sampler=$USES_BATCH_SAMPLER \
train_config.dataset.transforms.set.random_patch=$USES_RANDOM_PATCH \
train_config.dataset.transforms.set.random_rotate=$USES_RANDOM_ROTATE \
train_config.dataset.transforms.set.random_shift=$USES_RANDOM_SHIFT \
train_config.dataset.transforms.H_pad=$H_PAD \
train_config.dataset.transforms.W_pad=$W_PAD \
train_config.loss.grad_accum_steps=10 \
"train_config.loss.focal.alpha=$FOCAL_ALPHA" \
train_config.loss.focal.gamma=$FOCAL_GAMMA \
train_config.optim.lr=0.0003 \
train_config.optim.fused=false \
train_config.misc.monitors_dynamics=false \
train_config.misc.compiles_model=false \
train_config.misc.max_eval_iter=40 \
train_config.misc.data_dump_on=false \
train_config.lr_scheduler.warmup_iterations=10 \
train_config.lr_scheduler.total_iterations=3200 \
train_config.logging.prefix=$JOB \
train_config.dist.dtype=bfloat16

base_command="mpirun -n $NUM_MPI_TASKS python train.fsdp.py experiments/yaml/$JOB.yaml"
final_command="OMP_NUM_THREADS=1 "

if [ $RUNS_NSYS -eq 1 ]; then
    final_command+="nsys profile -w true -t cuda,mpi --mpi-impl=openmpi --wait primary -o /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/nsight_reports/$JOB.profile -f true --cudabacktrace=true -x true "
fi
final_command+="$base_command"

eval $final_command
