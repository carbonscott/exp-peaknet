#!/bin/bash

RUNS_NSYS=0
## NUM_MPI_TASKS=4
NNODES=10
QOS=debug
WALLTIME="02:00"

JOB=summit-distill-nano-2.0
BATCH_SIZE=1
H_PAD=1920
W_PAD=1920
NUM_WORKERS=1
USES_PAD=true
USES_POLAR_CENTER_CROP=false
USES_BATCH_SAMPLER=false
USES_RANDOM_PATCH=false
USES_RANDOM_ROTATE=false
USES_RANDOM_SHIFT=false
PATH_CHKPT_PREV=null
PREEEMPT_CHKPT_SAVING_ITERATIONS=10
CHKPT_SAVING_ITERATIONS=20
GRAD_ACCUM_STEPS=10
WARMUP=20
SHARDING_STAGE="zero3"

# [KNOWLEDGE DISTILLATION]
TEMPERATURE=2.0
FOCAL_ALPHA="[0.25, 0.75]"
FOCAL_GAMMA=2
LAM_MSE=0.4
LAM_KL=0.4
LAM_FOCAL=0.2
EMA_MOMENTUM=0.9

# [DATASET]
## PATH_TRAIN="distill/mfxl1025422.train.csv"
## PATH_EVAL="distill/mfxl1025422.eval.csv"
PATH_TRAIN="distill/mfxl1027522.train.csv"
PATH_EVAL="distill/mfxl1027522.eval.csv"

PREEMPT_ROOT="preempt"
mkdir -p $PREEMPT_ROOT
PREEMPT_METADATA_PATH="$PREEMPT_ROOT/$JOB"

SEG_SIZE=$((BATCH_SIZE * 60))

python launch_job.distill.py \
job=$JOB \
auto_submit=false \
bsub_config.trainer=train.distill.py \
bsub_config.num_gpus_per_node=6 \
bsub_config.num_cpus_per_rs=2 \
bsub_config.num_nodes=$NNODES \
bsub_config.walltime=$WALLTIME \
bsub_config.qos=$QOS \
distill_config.checkpoint.prefix=$JOB \
distill_config.checkpoint.path_chkpt_prev=$PATH_CHKPT_PREV \
distill_config.checkpoint.state_dict_type=full \
distill_config.checkpoint.preempt_metadata_path=$PREEMPT_METADATA_PATH \
distill_config.checkpoint.preempt_chkpt_saving_iterations=$PREEEMPT_CHKPT_SAVING_ITERATIONS \
distill_config.checkpoint.chkpt_saving_iterations=$CHKPT_SAVING_ITERATIONS \
"distill_config.dataset.path_train=$PATH_TRAIN" \
"distill_config.dataset.path_eval=$PATH_EVAL" \
distill_config.dataset.num_workers=$NUM_WORKERS \
distill_config.dataset.prefetch_factor=10 \
distill_config.dataset.pin_memory=true \
distill_config.dataset.seg_size=$SEG_SIZE \
distill_config.dataset.batch_size=$BATCH_SIZE \
distill_config.dataset.transforms.set.pad=$USES_PAD \
distill_config.dataset.transforms.set.polar_center_crop=$USES_POLAR_CENTER_CROP \
distill_config.dataset.transforms.set.batch_sampler=$USES_BATCH_SAMPLER \
distill_config.dataset.transforms.set.random_patch=$USES_RANDOM_PATCH \
distill_config.dataset.transforms.set.random_rotate=$USES_RANDOM_ROTATE \
distill_config.dataset.transforms.set.random_shift=$USES_RANDOM_SHIFT \
distill_config.dataset.transforms.H_pad=$H_PAD \
distill_config.dataset.transforms.W_pad=$W_PAD \
distill_config.model.backbone.hf_config.image_size=$H_PAD \
"distill_config.model.backbone.hf_config.hidden_sizes=[80, 160, 320, 640]" \
"distill_config.model.backbone.hf_config.depths=[2, 2, 8, 2]" \
"distill_config.model.backbone.hf_config.patch_size=4" \
"distill_config.model.bifpn.block.num_features=512" \
"distill_config.model.bifpn.num_blocks=4" \
distill_config.model.seg_head.uses_learned_upsample=true \
distill_config.loss.grad_accum_steps=$GRAD_ACCUM_STEPS \
"distill_config.loss.focal.alpha=$FOCAL_ALPHA" \
distill_config.loss.focal.gamma=$FOCAL_GAMMA \
distill_config.loss.temperature=$TEMPERATURE \
distill_config.loss.lam_mse=$LAM_MSE \
distill_config.loss.lam_kl=$LAM_KL \
distill_config.loss.lam_kl=$LAM_FOCAL \
distill_config.loss.ema_momentum=$EMA_MOMENTUM \
distill_config.optim.lr=0.0003 \
distill_config.optim.fused=false \
distill_config.misc.monitors_dynamics=false \
distill_config.misc.compiles_model=false \
distill_config.misc.max_eval_iter=40 \
distill_config.misc.max_epochs=200 \
"distill_config.misc.sharding_stage=$SHARDING_STAGE" \
distill_config.misc.data_dump_on=false \
distill_config.lr_scheduler.warmup_iterations=$WARMUP \
distill_config.lr_scheduler.total_iterations=3200 \
distill_config.logging.prefix=$JOB \
distill_config.dist.dtype=float16

## ## base_command="mpirun -n $NUM_MPI_TASKS --map-by ppr:${NUM_MPI_TASKS}:node --bind-to none python train.distill.py experiments/yaml/$JOB.yaml"
## base_command="mpirun -n $NUM_MPI_TASKS `which python` train.distill.py experiments/yaml/$JOB.yaml"
## final_command="OMP_NUM_THREADS=1 "
## 
## if [ $RUNS_NSYS -eq 1 ]; then
##     final_command+="nsys profile -w true -t cuda,mpi --mpi-impl=openmpi --wait primary -o /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/nsight_reports/$JOB.profile -f true --cudabacktrace=true -x true "
## fi
## final_command+="$base_command"
## 
## eval $final_command
