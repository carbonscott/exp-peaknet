# PeakNet Project

A deep learning project for peak detection and analysis with multi-facility HPC job launching capabilities.

## 🚀 Quick Start with Unified Launcher

**Recommended for all new users:**

```bash
# Generate job scripts for ALL HPC facilities automatically
python launch_unified_hydra.py job=my-experiment train_config=hiera resource_configs=hiera

# Submit to your current facility:
sbatch experiments/jobs/my-experiment.s3df.sbatch      # SLAC S3DF
sbatch experiments/jobs/my-experiment.nersc.sbatch     # NERSC Perlmutter  
bsub experiments/jobs/my-experiment.summit.bsub        # ORNL Summit
```

**Benefits:**
- ✅ **Future-proof**: Works on any HPC facility by just adding template files
- ✅ **No facility detection**: Generates all scripts, you pick what works
- ✅ **Smart naming**: Clear `<job>.<facility>.<scheduler>` format
- ✅ **Backward compatible**: All existing scripts still work

## 📖 Documentation

- **[Hydra Config Guide](hydra_config/README.md)** - Complete configuration system documentation
- **[Migration Guide](#migration-guide)** - How to upgrade from legacy launchers
- **[Examples](#examples)** - Common usage patterns

## 🏗️ Job Launcher Comparison

### 🆕 **NEW: Hydra-Native Unified Launcher** (Recommended)
```bash
python launch_unified_hydra.py job=my-job train_config=hiera resource_configs=hiera
```
- **Hydra-native**: Direct integration with Hydra configuration system
- **One launcher for all facilities** (SLAC, NERSC, ORNL, etc.)
- **Auto-generates scripts** for all supported facilities
- **Future-proof**: Just add template files for new facilities

### 🔧 **LEGACY: Facility-Specific Launchers** (Still supported)
```bash
python launch_sbatch_job.py job=my-job sbatch_config=hiera train_config=hiera  # SLURM only
python launch_job.py job=my-job bsub_config=base train_config=hiera           # LSF only
```
- **Facility-specific**: Need different launcher for each HPC system
- **Manual facility detection**: User must know which launcher to use

## 🔄 Migration Guide

### For New Users
**Just use the unified launcher!** It's simpler and works everywhere.

### For Existing Users
Your old scripts still work, but consider migrating:

**Old way:**
```bash
python launch_sbatch_job.py job=my-job sbatch_config=hiera train_config=hiera
```

**New way:**
```bash
python launch_unified_hydra.py job=my-job resource_configs=hiera train_config=hiera
```

**Key changes:**
- `sbatch_config=hiera` → `resource_configs=hiera` (clearer naming)
- Hydra-native launcher with direct config integration
- Single launcher instead of facility-specific ones
- All job scripts generated automatically

## 💡 Examples

### Basic Training Job
```bash
# Generate scripts for all facilities
python launch_unified_hydra.py job=my-training train_config=base resource_configs=base

# Auto-submit to specific facility  
python launch_unified_hydra.py job=my-training auto_submit=true target_facility=s3df
```

### Hiera Model Training
```bash
# Use the adapted Hiera training script
./run_hiera_ddp_training_unified.sh

# Or generate scripts only
export GENERATE_ONLY=true
./run_hiera_ddp_training_unified.sh
```

### Custom Resource Configuration
```bash
# Create your own resource config in hydra_config/resource_configs/my_config.yaml
python launch_unified_hydra.py job=my-job resource_configs=my_config train_config=hiera
```

### Hiera-Huge Production Training
```bash
# Use the complete Hiera-Huge production configuration
python launch_unified_hydra.py job=hiera-huge-production resource_configs=hiera train_config=hiera_huge
```

### Large-Scale Throughput Testing
```bash
# Test training throughput on 100 nodes (1000 GPUs)
python launch_unified_hydra.py job=throughput-test resource_configs=throughput_100node train_config=throughput_test
```

## 🏛️ Supported HPC Facilities

| Facility | Scheduler | Status | Template |
|----------|-----------|--------|----------|
| **SLAC S3DF** | SLURM | ✅ Ready | `s3df.sbatch` |
| **NERSC Perlmutter** | SLURM | ✅ Ready | `nersc.sbatch` |  
| **ORNL Summit** | LSF | ✅ Ready | `summit.bsub` |
| **ORNL Frontier** | SLURM | 🚧 Add template | `frontier.sbatch` |
| **ANL Aurora** | PBS | 🚧 Add template | `aurora.pbs` |

### Adding New Facilities
To add support for a new HPC facility:

1. **Create template**: `hydra_config/templates/new_facility.sbatch`
2. **Create scheduler config**: `hydra_config/scheduler_configs/new_facility.yaml`
3. **Done!** The unified launcher auto-discovers and generates scripts

No code changes needed - just configuration files.

## 🔧 Configuration System

The project uses a hierarchical configuration system:

```
hydra_config/
├── templates/              # Job script templates for each facility
├── scheduler_configs/      # Facility-specific settings (accounts, partitions)  
├── resource_configs/       # Compute requirements (GPUs, walltime, memory)
├── train_config/          # Training parameters and model configs
└── distill_config/        # Knowledge distillation configs
```

**Key concepts:**
- **Templates**: Job script skeletons with placeholders
- **Scheduler configs**: Facility-specific (account IDs, partitions, modules)
- **Resource configs**: Compute requirements (independent of facility)
- **Train configs**: Model architecture and training parameters

## 🧪 Testing

Validate that the unified launcher works:

```bash
python test_unified_launcher.py
```

## 🆘 Troubleshooting

**Q: My old scripts stopped working**  
A: Legacy scripts are still supported. Check the [hydra config README](hydra_config/README.md) for details.

**Q: How do I know which facility I'm on?**  
A: Generate all scripts with the unified launcher, then submit the one that works.

**Q: Can I customize job templates?**  
A: Yes! Edit files in `hydra_config/templates/` or create new ones.

**Q: How do I add custom resource configurations?**  
A: Create new YAML files in `hydra_config/resource_configs/` and reference them with `resource_configs=your_config`.

## 📋 Requirements

- Python 3.7+
- PyTorch
- Hydra
- OmegaConf
- Jinja2

## 🤝 Contributing

When adding support for new HPC facilities:
1. Add template in `hydra_config/templates/`
2. Add scheduler config in `hydra_config/scheduler_configs/`
3. Test with `python launch_unified_hydra.py`
4. Update this README

## 📄 License

[Add your license information here]