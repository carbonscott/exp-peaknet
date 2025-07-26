#!/usr/bin/env python3
"""
Simple validation script for the unified job launcher.

Tests that the launcher can:
1. Auto-discover templates
2. Generate job scripts for all facilities 
3. Create valid YAML configs
4. Handle different resource configurations
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import subprocess


def test_basic_functionality():
    """Test basic launcher functionality"""
    print("🧪 Testing basic launcher functionality...")
    
    # Test basic job generation
    cmd = [
        "python", "launch_unified.py",
        "job=test-validation",
        "train_config=base",
        "resource_configs=base", 
        "auto_submit=false"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Basic launcher test failed:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    # Check that job scripts were generated
    expected_files = [
        "experiments/jobs/test-validation.s3df.sbatch",
        "experiments/jobs/test-validation.nersc.sbatch", 
        "experiments/jobs/test-validation.summit.bsub",
        "experiments/yaml/test-validation.yaml"
    ]
    
    for file_path in expected_files:
        if not os.path.exists(file_path):
            print(f"❌ Expected file not generated: {file_path}")
            return False
    
    print("✅ Basic functionality test passed!")
    return True


def test_template_discovery():
    """Test that templates are discovered correctly"""
    print("🧪 Testing template discovery...")
    
    template_dir = Path("hydra_config/templates")
    if not template_dir.exists():
        print(f"❌ Template directory not found: {template_dir}")
        return False
    
    expected_templates = ["s3df.sbatch", "nersc.sbatch", "summit.bsub"]
    
    for template in expected_templates:
        template_path = template_dir / template
        if not template_path.exists():
            print(f"❌ Expected template not found: {template_path}")
            return False
    
    print("✅ Template discovery test passed!")
    return True


def test_config_structure():
    """Test that config structure is correct"""
    print("🧪 Testing config structure...")
    
    required_dirs = [
        "hydra_config/templates",
        "hydra_config/scheduler_configs", 
        "hydra_config/resource_configs",
        "hydra_config/train_config"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Required directory not found: {dir_path}")
            return False
    
    # Check that base configs exist
    required_files = [
        "hydra_config/base.yaml",
        "hydra_config/resource_configs/base.yaml",
        "hydra_config/train_config/base.yaml",
        "hydra_config/scheduler_configs/s3df.yaml"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required config file not found: {file_path}")
            return False
    
    print("✅ Config structure test passed!")
    return True


def test_hiera_config():
    """Test Hiera-specific configuration"""
    print("🧪 Testing Hiera configuration...")
    
    cmd = [
        "python", "launch_unified.py",
        "job=test-hiera-validation", 
        "train_config=hiera",
        "resource_configs=hiera",
        "auto_submit=false"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Hiera config test failed:")
        print(f"STDERR: {result.stderr}")
        return False
    
    # Check that Hiera-specific job script was generated with correct settings
    s3df_script = "experiments/jobs/test-hiera-validation.s3df.sbatch"
    if os.path.exists(s3df_script):
        with open(s3df_script, 'r') as f:
            content = f.read()
            # Check for Hiera-specific settings (48hr walltime, 10 GPUs)
            if "--time 48:00:00" not in content:
                print("❌ Hiera walltime not found in generated script")
                return False
            if "--gres=gpu:10" not in content:
                print("❌ Hiera GPU count not found in generated script") 
                return False
    
    print("✅ Hiera configuration test passed!")
    return True


def test_generated_script_validity():
    """Test that generated scripts have valid syntax"""
    print("🧪 Testing generated script validity...")
    
    script_files = [
        "experiments/jobs/test-validation.s3df.sbatch",
        "experiments/jobs/test-validation.nersc.sbatch",
        "experiments/jobs/test-validation.summit.bsub"
    ]
    
    for script_file in script_files:
        if os.path.exists(script_file):
            with open(script_file, 'r') as f:
                content = f.read()
                
                # Basic validity checks
                if not content.startswith('#!/bin/bash'):
                    print(f"❌ Script missing shebang: {script_file}")
                    return False
                
                # Check for scheduler directives
                if '.sbatch' in script_file and '#SBATCH' not in content:
                    print(f"❌ SBATCH script missing SBATCH directives: {script_file}")
                    return False
                    
                if '.bsub' in script_file and '#BSUB' not in content:
                    print(f"❌ BSUB script missing BSUB directives: {script_file}")
                    return False
                
                # Check for job execution command
                if 'python train_hiera_seg.py' not in content and 'python train' not in content:
                    print(f"❌ Script missing training command: {script_file}")
                    return False
    
    print("✅ Generated script validity test passed!")
    return True


def cleanup_test_files():
    """Clean up generated test files"""
    print("🧹 Cleaning up test files...")
    
    test_files = [
        "experiments/jobs/test-validation.s3df.sbatch",
        "experiments/jobs/test-validation.nersc.sbatch", 
        "experiments/jobs/test-validation.summit.bsub",
        "experiments/yaml/test-validation.yaml",
        "experiments/jobs/test-hiera-validation.s3df.sbatch",
        "experiments/jobs/test-hiera-validation.nersc.sbatch",
        "experiments/jobs/test-hiera-validation.summit.bsub", 
        "experiments/yaml/test-hiera-validation.yaml"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   Removed: {file_path}")


def main():
    """Run all validation tests"""
    print("🚀 Unified Launcher Validation Tests")
    print("=" * 50)
    
    tests = [
        test_template_discovery,
        test_config_structure, 
        test_basic_functionality,
        test_hiera_config,
        test_generated_script_validity
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error in {test.__name__}: {e}")
    
    print("\\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Unified launcher is working correctly.")
        cleanup_test_files()
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())