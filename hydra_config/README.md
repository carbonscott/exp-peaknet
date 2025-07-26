# Hydra Configuration Directory Structure

This directory contains configuration files for the project's job launching system. There are **two systems** in use:

## 🆕 **NEW: Unified Multi-Facility Launcher** (Recommended)

Use `launch_unified.py` for new projects. It supports all HPC facilities automatically.

### Directory Structure:
```
hydra_config/
├── templates/              # Job script templates for each facility
│   ├── s3df.sbatch        # SLAC S3DF (SLURM)
│   ├── nersc.sbatch       # NERSC Perlmutter (SLURM)
│   └── summit.bsub        # ORNL Summit (LSF)
│
├── scheduler_configs/      # Facility-specific settings
│   ├── s3df.yaml          # SLAC account, partition, launcher command
│   ├── nersc.yaml         # NERSC account, QoS, conda environment
│   └── summit.yaml        # ORNL project, resource allocation
│
├── resource_configs/       # Compute resource specifications
│   ├── base.yaml          # Default: 12hr, 10 GPUs, ada partition
│   └── hiera.yaml         # Hiera training: 48hr, 10 GPUs, specialized config
│
├── train_config/          # Training configurations
│   ├── base.yaml          # Standard training parameters
│   └── hiera.yaml         # Hiera model architecture & training settings
│
└── distill_config/        # Knowledge distillation configurations
    └── base.yaml          # Distillation parameters
```

### Usage:
```bash
# Generate job scripts for ALL facilities
python launch_unified.py job=my-job resource_config=hiera train_config=hiera

# Auto-submit to specific facility  
python launch_unified.py job=my-job auto_submit=true target_facility=s3df

# Then submit the appropriate script for your current facility:
sbatch experiments/jobs/my-job.s3df.sbatch      # SLAC S3DF
sbatch experiments/jobs/my-job.nersc.sbatch     # NERSC
bsub experiments/jobs/my-job.summit.bsub        # ORNL Summit
```

### Benefits:
- ✅ **Future-proof**: Just add new template files for new facilities
- ✅ **No facility detection**: Generate all scripts, pick what works
- ✅ **Clear separation**: Resource specs vs facility-specific settings
- ✅ **Smart naming**: `<job>.<facility>.<scheduler>` format

---

## 🔧 **LEGACY: Facility-Specific Launchers** (Backward Compatibility)

⚠️ **DEPRECATED**: These are kept for backward compatibility with existing scripts.

### Directory Structure:
```
hydra_config/
├── bsub_config/           # DEPRECATED: LSF/ORNL Summit configs
│   ├── base.yaml         # 🔧 Legacy LSF settings
│   └── template.bsub     # 🔧 Legacy LSF job template
│
└── sbatch_config/         # DEPRECATED: SLURM configs  
    ├── base.yaml         # 🔧 Legacy SLURM settings
    ├── hiera.yaml        # 🔧 Legacy Hiera SLURM settings
    ├── template.sbatch   # 🔧 Legacy S3DF SLURM template
    └── template.nersc.sbatch # 🔧 Legacy NERSC SLURM template
```

### Legacy Launchers (still work, but deprecated):
- `launch_job.py` - LSF/Summit launcher
- `launch_sbatch_job.py` - SLURM/S3DF launcher  
- `launch_job.distill.py` - Distillation launcher
- `launch_job.slurm.py` - Generic SLURM launcher

---

## 🚀 **Migration Guide**

### For New Users:
**Use the unified launcher!** It's simpler and works everywhere:
```bash
python launch_unified.py job=my-job train_config=hiera resource_config=hiera
```

### For Existing Users:
Your old scripts still work, but consider migrating:

**Old way:**
```bash
python launch_sbatch_job.py job=my-job sbatch_config=hiera train_config=hiera
```

**New way:**
```bash  
python launch_unified.py job=my-job resource_config=hiera train_config=hiera
```

### Key Changes:
- `sbatch_config=hiera` → `resource_config=hiera` (clearer naming)
- Single launcher for all facilities vs facility-specific launchers
- Job scripts generated for all facilities automatically

---

## 📂 **Configuration Details**

### Resource Configs (`resource_configs/`)
Define **compute requirements** (independent of facility):
- Walltime, number of GPUs, memory
- Training script, batch size
- Model architecture parameters

### Scheduler Configs (`scheduler_configs/`)  
Define **facility-specific settings**:
- Account/project IDs
- Partition/queue names
- Module loads, proxy settings
- Job launcher commands (`mpirun`, `srun`, `jsrun`)

### Templates (`templates/`)
Job script templates with Jinja2 placeholders:
- Scheduler directives (`#SBATCH`, `#BSUB`)
- Environment setup
- Job execution commands

---

## 🔍 **Adding New Facilities**

To add support for a new HPC facility:

1. **Create template**: `hydra_config/templates/new_facility.sbatch`
2. **Create scheduler config**: `hydra_config/scheduler_configs/new_facility.yaml`  
3. **Done!** The unified launcher auto-discovers and generates scripts

No code changes needed - just configuration files.

---

## 🆘 **Troubleshooting**

**Q: My old scripts stopped working**  
A: Legacy scripts are still supported. Check that you're using the correct paths in `base.yaml`.

**Q: How do I know which facility I'm on?**  
A: Generate all scripts, then submit the one that works. The unified launcher makes this easy.

**Q: Can I customize job templates?**  
A: Yes! Edit files in `templates/` or `scheduler_configs/` to customize for your needs.

**Q: How do I add custom resource configurations?**  
A: Create new YAML files in `resource_configs/` and reference them with `resource_config=your_config`.

---

**📧 Questions?** Check the launcher source code or ask the maintainers.