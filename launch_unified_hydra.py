#!/usr/bin/env python3
"""
Hydra-Native Unified Job Launcher

Usage:
    python launch_unified_hydra.py job=my_job train_config=hiera auto_submit=false

Key Changes from Original:
- Generates Hydra commands instead of YAML files
- Job scripts call training script directly with Hydra overrides
- Much simpler - no intermediate YAML generation needed!
"""

import os
import sys
import glob
import yaml
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
from jinja2 import Environment, FileSystemLoader


def discover_templates(template_dir="hydra_config/templates"):
    """Auto-discover all template files and return facility info"""
    template_files = glob.glob(f"{template_dir}/*.*")
    facilities = []

    for template_path in template_files:
        template_file = os.path.basename(template_path)
        # Extract facility name and scheduler from filename
        name_parts = template_file.split('.')
        if len(name_parts) >= 2:
            facility = name_parts[0]
            scheduler = name_parts[1]
            facilities.append({
                'facility': facility,
                'scheduler': scheduler, 
                'template_path': template_path,
                'template_file': template_file
            })

    return facilities


def get_facility_config(cfg: DictConfig, facility: str):
    """
    Get facility-specific configuration using Hydra's native merging.
    
    This replaces the manual config loading and merging with Hydra's
    built-in configuration composition and override system.
    
    Priority order (highest to lowest):
    1. CLI overrides and resource_configs (cfg) 
    2. Scheduler configs (facility defaults)
    """
    
    # Load facility-specific scheduler config first (lower priority)
    scheduler_config_path = f"hydra_config/scheduler_configs/{facility}.yaml"
    if os.path.exists(scheduler_config_path):
        scheduler_cfg = OmegaConf.load(scheduler_config_path)
    else:
        print(f"Warning: No scheduler config found for {facility}")
        scheduler_cfg = OmegaConf.create({})
    
    # Start with scheduler defaults
    facility_config = scheduler_cfg.copy()
    
    # Flatten resource_configs into top-level (higher priority)
    if hasattr(cfg, 'resource_configs') and cfg.resource_configs:
        resource_flat = OmegaConf.to_container(cfg.resource_configs, resolve=True)
        # Remove internal Hydra fields
        if '_name' in resource_flat:
            del resource_flat['_name']
        facility_config.update(resource_flat)
    
    # Apply any other top-level overrides from cfg (highest priority)
    cfg_flat = OmegaConf.to_container(cfg, resolve=True)
    for key, value in cfg_flat.items():
        if key != 'resource_configs':  # Don't overwrite our flattened resource_configs
            facility_config[key] = value
    
    return OmegaConf.create(facility_config)


def validate_config(config_dict, facility):
    """Validate critical configuration parameters"""
    warnings = []
    errors = []
    
    # Check required fields
    if config_dict.get('num_nodes', 0) <= 0:
        errors.append(f"num_nodes must be > 0, got: {config_dict.get('num_nodes')}")
    
    if config_dict.get('num_tasks', 0) <= 0:
        errors.append(f"num_tasks must be > 0, got: {config_dict.get('num_tasks')}")
    
    # Facility-specific validations
    if facility == 'frontier' and config_dict.get('num_nodes', 0) > 0:
        num_tasks = config_dict.get('num_tasks', 0)
        num_nodes = config_dict.get('num_nodes', 1)
        
        if num_tasks % num_nodes != 0:
            warnings.append(f"Frontier: num_tasks ({num_tasks}) should be divisible by num_nodes ({num_nodes})")
        
        if num_tasks // num_nodes > 8:
            warnings.append(f"Frontier: More than 8 tasks per node may exceed GPU capacity")
    
    return warnings, errors


# DEPRECATED: Keep for backward compatibility if needed
def load_scheduler_config(facility, config_dir="hydra_config/scheduler_configs"):
    """DEPRECATED: Load facility-specific scheduler configuration"""
    config_path = f"{config_dir}/{facility}.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Warning: No scheduler config found for {facility} at {config_path}")
        return {}


def load_resource_config(resource_name, config_dir="hydra_config/resource_configs"):
    """DEPRECATED: Load resource-specific configuration"""
    config_path = f"{config_dir}/{resource_name}.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Warning: No resource config found for {resource_name} at {config_path}")
        return {}


def build_hydra_command(cfg, base_trainer="train_hydra_demo.py"):
    """
    BUILD HYDRA COMMAND INSTEAD OF YAML FILE
    
    This is the key change! Instead of generating a YAML file,
    we build a Hydra command with all the overrides.
    """
    
    # Start with base command
    cmd_parts = [f"python {base_trainer}"]
    
    # Add all the overrides that would normally go in YAML
    overrides = []
    
    # Job name
    overrides.append(f"job={cfg.job}")
    
    # Resource and train configs (these are the main config groups)
    if hasattr(cfg, 'resource_configs') and hasattr(cfg.resource_configs, '_name'):
        overrides.append(f"resource_configs={cfg.resource_configs._name}")
    if hasattr(cfg, 'train_config') and hasattr(cfg.train_config, '_name'):
        overrides.append(f"train_config={cfg.train_config._name}")
    
    # Any other direct overrides from the launch command
    # These would be things like checkpoint.path_chkpt_prev=path
    # We can add these dynamically based on command line args
    
    # Join all parts
    if overrides:
        cmd_parts.append(" ".join(overrides))
    
    return " ".join(cmd_parts)


def render_job_script(template_path, config_data):
    """Render job script using Jinja2 template"""
    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    return template.render(**config_data)


@hydra.main(config_path="hydra_config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    HYDRA-NATIVE LAUNCHER
    
    Key difference: We generate Hydra commands, not YAML files!
    """

    print("🚀 Hydra-Native Unified Job Launcher")
    print("=" * 50)

    # ----------------------------------------------------------------------- #
    #  Setup shared values
    # ----------------------------------------------------------------------- #
    job = cfg.job
    trainer = cfg.get('trainer', 'train_hydra_demo.py')

    # Create output directories (only for job scripts now!)
    os.makedirs("experiments/jobs", exist_ok=True)

    # ----------------------------------------------------------------------- #
    #  Build Hydra command instead of YAML file
    # ----------------------------------------------------------------------- #
    hydra_command = build_hydra_command(cfg, trainer)
    print(f"🎯 Generated Hydra command:")
    print(f"   {hydra_command}")
    print()

    # ----------------------------------------------------------------------- #
    #  Auto-discover templates and generate job scripts  
    # ----------------------------------------------------------------------- #
    facilities = discover_templates()

    if not facilities:
        print("❌ No templates found in hydra_config/templates/")
        return

    print(f"🔍 Discovered {len(facilities)} facilities:")
    for facility_info in facilities:
        print(f"   - {facility_info['facility']}.{facility_info['scheduler']}")
    print()

    generated_scripts = []

    for facility_info in facilities:
        facility = facility_info['facility']
        scheduler = facility_info['scheduler']
        template_path = facility_info['template_path']

        print(f"🏗️  Generating {facility} job script...")

        # Get facility-specific config using proper Hydra merging
        facility_config = get_facility_config(cfg, facility)
        
        # Convert OmegaConf to dict for Jinja template compatibility
        merged_config = OmegaConf.to_container(facility_config, resolve=True)
        
        # Validate configuration
        warnings, errors = validate_config(merged_config, facility)
        
        if errors:
            print(f"   ❌ Configuration errors for {facility}:")
            for error in errors:
                print(f"      - {error}")
            continue  # Skip this facility
        
        if warnings:
            print(f"   ⚠️  Configuration warnings for {facility}:")
            for warning in warnings:
                print(f"      - {warning}")
        
        # Add launcher-specific values
        merged_config.update({
            'job': job,
            'hydra_command': hydra_command,  # NEW: Pass the Hydra command
            'trainer': trainer
        })

        # Render job script
        try:
            rendered_script = render_job_script(template_path, merged_config)

            # Save job script
            job_filename = f"{job}.{facility}.{scheduler}"
            job_path = f"experiments/jobs/{job_filename}"

            with open(job_path, 'w') as f:
                f.write(rendered_script)
                f.write("\n")
                f.write(f"# Generated by: python {' '.join(sys.argv)}\n")

            generated_scripts.append(job_path)
            print(f"   ✅ {job_path}")

        except Exception as e:
            print(f"   ❌ Failed to generate {facility} script: {e}")

    # ----------------------------------------------------------------------- #
    #  Summary and submission
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 50)
    print(f"📦 Generated {len(generated_scripts)} job scripts:")
    for script in generated_scripts:
        print(f"   {script}")

    print(f"\n🎯 Each script runs: {hydra_command}")

    # Handle auto-submission (same logic as before)
    if cfg.get('auto_submit', False):
        target_facility = cfg.get('target_facility', None)
        if target_facility:
            target_scripts = [s for s in generated_scripts if f".{target_facility}." in s]
            if target_scripts:
                script = target_scripts[0]
                if 'sbatch' in script:
                    cmd = f"sbatch {script}"
                elif 'bsub' in script:
                    cmd = f"bsub {script}"
                else:
                    print(f"❌ Unknown scheduler type for {script}")
                    return

                print(f"🚀 Auto-submitting: {cmd}")
                os.system(cmd)
            else:
                print(f"❌ No scripts found for target facility: {target_facility}")
        else:
            print("❌ auto_submit=true requires target_facility to be specified")
    else:
        print("\n💡 To submit a job, run:")
        for script in generated_scripts:
            if 'sbatch' in script:
                print(f"   sbatch {script}")
            elif 'bsub' in script:
                print(f"   bsub {script}")


if __name__ == "__main__":
    main()