#!/bin/bash
#
# Basic Usage Examples for Unified Job Launcher
#

echo "🚀 Unified Launcher - Basic Usage Examples"
echo "=========================================="

# Example 1: Generate job scripts for all facilities
echo "📦 Example 1: Generate scripts for all facilities"
python launch_unified.py \
    job=my-basic-experiment \
    train_config=base \
    resource_configs=base \
    auto_submit=false

echo ""
echo "✅ Generated scripts:"
echo "   - experiments/jobs/my-basic-experiment.s3df.sbatch"
echo "   - experiments/jobs/my-basic-experiment.nersc.sbatch" 
echo "   - experiments/jobs/my-basic-experiment.summit.bsub"
echo ""

# Example 2: Auto-submit to specific facility
echo "📤 Example 2: Auto-submit to SLAC S3DF"
python launch_unified.py \
    job=my-auto-submit-experiment \
    train_config=base \
    resource_configs=base \
    auto_submit=true \
    target_facility=s3df

echo ""

# Example 3: Hiera model training with custom settings
echo "🧠 Example 3: Hiera model training"
python launch_unified.py \
    job=hiera-training \
    train_config=hiera \
    resource_configs=hiera \
    train_config.model.hiera.embed_dim=256 \
    train_config.optim.lr=0.0005 \
    auto_submit=false

echo ""

# Example 4: Custom resource configuration
echo "⚙️ Example 4: Custom walltime and GPU count"
python launch_unified.py \
    job=custom-resources \
    train_config=base \
    resource_configs=base \
    resource_configs.walltime="24:00:00" \
    resource_configs.num_gpus_per_node=8 \
    auto_submit=false

echo ""
echo "💡 To submit any generated script:"
echo "   sbatch experiments/jobs/<job-name>.s3df.sbatch      # SLAC S3DF"
echo "   sbatch experiments/jobs/<job-name>.nersc.sbatch     # NERSC"
echo "   bsub experiments/jobs/<job-name>.summit.bsub        # ORNL Summit"