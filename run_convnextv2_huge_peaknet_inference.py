#!/usr/bin/env python3
"""
High-Performance ConvNextV2 Huge PeakNet Inference Script with CUDA Graph Support

This script provides optimized inference for ConvNextV2 huge based PeakNet models
using a high-performance pipeline with double buffering, mixed precision, and
torch compilation for maximum GPU utilization.

Features:
- YAML-based configuration (uses peaknet-673m.yaml)
- High-performance inference pipeline (97-99% GPU utilization)
- CUDA graph optimization for 10-20% speedup (reduce-overhead/max-autotune modes)
- Memory-optimized compilation for large models
- Automatic batch size optimization
- Mixed precision support (bfloat16)
- Torch compilation support with smart defaults
- Built-in benchmarking and profiling
- NVTX markers for nsys profiling

Usage Examples:

    # Maximum Performance with Caching (CUDA graphs + cache artifacts)
    python run_convnextv2_huge_peaknet_inference.py \
        --yaml-path peaknet-673m.yaml \
        --compile-mode reduce-overhead \
        --batch-size 16 --benchmark \
        --cache-artifacts ./peaknet_cache.bin

    # Memory-Constrained Systems with Caching
    python run_convnextv2_huge_peaknet_inference.py \
        --yaml-path peaknet-673m.yaml \
        --compile-mode reduce-overhead \
        --warmup-batch-size 1 \
        --batch-size 16 --benchmark \
        --cache-artifacts ./peaknet_memory_cache.bin

    # Custom Cache Directory
    python run_convnextv2_huge_peaknet_inference.py \
        --yaml-path peaknet-673m.yaml \
        --compile-mode max-autotune \
        --batch-size 16 \
        --cache-dir /fast/ssd/torch_cache \
        --cache-artifacts ./peaknet_max_cache.bin

    # Basic Usage (no compilation)
    python run_convnextv2_huge_peaknet_inference.py \
        --yaml-path peaknet-673m.yaml \
        --batch-size 8

Compilation Trade-offs (controlled by --warmup-batch-size):
- None (default): CUDA graphs enabled, 10-20% speedup, standard compilation memory
- 1: Memory-optimized compilation, CUDA graphs disabled, prevents OOM on large models
- Pipeline automatically adjusts warmup samples (1000+) when CUDA graphs are enabled

Caching Benefits (--cache-artifacts):
- First run: Compiles model and saves cache artifacts (full warmup time)
- Subsequent runs: Loads cache and skips most warmup (near-instant startup)
- Portable: Cache artifacts work across machines with same PyTorch/CUDA versions
- Production-ready: Pre-compile models and deploy with cached artifacts
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

# PeakNet imports
from peaknet.modeling.convnextv2_bifpn_net import (
    PeakNet, PeakNetConfig, SegHeadConfig
)
from peaknet.modeling.bifpn_config import (
    BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
)
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config

# Inference pipeline imports
from inference_pipeline import (
    DoubleBufferedPipelineBase,
    ComputeWorkload,
    PipelineConfig,
    PipelineBenchmark
)
from inference_pipeline.utils.memory import generate_test_tensors

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_peaknet_from_yaml(yaml_path: str, weights_path: Optional[str] = None) -> PeakNet:
    """
    Load PeakNet model from YAML configuration file.

    Args:
        yaml_path: Path to YAML configuration file
        weights_path: Optional path to pre-trained weights

    Returns:
        Configured PeakNet model
    """
    logger.info(f"Loading PeakNet configuration from {yaml_path}")

    # Load YAML configuration
    config = OmegaConf.load(yaml_path)
    model_params = config.get("model")

    if model_params is None:
        raise ValueError(f"No 'model' section found in {yaml_path}")

    # Extract configuration parameters
    backbone_params = model_params.get("backbone", {})
    hf_model_config = backbone_params.get("hf_config", {})
    bifpn_params = model_params.get("bifpn", {})
    bifpn_block_params = bifpn_params.get("block", {})
    bifpn_block_bn_params = bifpn_block_params.get("bn", {})
    bifpn_block_fusion_params = bifpn_block_params.get("fusion", {})
    seghead_params = model_params.get("seg_head", {})

    # Build model configuration objects
    # Convert OmegaConf to regular dicts to avoid issues
    hf_model_config_dict = OmegaConf.to_container(hf_model_config, resolve=True)
    backbone_config = ConvNextV2Config(**hf_model_config_dict)

    # BiFPN configuration
    bifpn_block_bn_params_dict = OmegaConf.to_container(bifpn_block_bn_params, resolve=True)
    bifpn_block_fusion_params_dict = OmegaConf.to_container(bifpn_block_fusion_params, resolve=True)
    bifpn_block_params_dict = OmegaConf.to_container(bifpn_block_params, resolve=True)
    bifpn_params_dict = OmegaConf.to_container(bifpn_params, resolve=True)

    bifpn_block_params_dict["bn"] = BNConfig(**bifpn_block_bn_params_dict)
    bifpn_block_params_dict["fusion"] = FusionConfig(**bifpn_block_fusion_params_dict)
    bifpn_params_dict["block"] = BiFPNBlockConfig(**bifpn_block_params_dict)
    bifpn_config = BiFPNConfig(**bifpn_params_dict)

    # Segmentation head configuration
    seghead_params_dict = OmegaConf.to_container(seghead_params, resolve=True)
    seghead_config = SegHeadConfig(**seghead_params_dict)

    # Create PeakNet configuration
    peaknet_config = PeakNetConfig(
        backbone=backbone_config,
        bifpn=bifpn_config,
        seg_head=seghead_config,
    )

    # Create model
    model = PeakNet(peaknet_config)
    model.init_weights()

    # Load weights if provided
    if weights_path and os.path.exists(weights_path):
        logger.info(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"PeakNet model loaded: {num_params/1e6:.1f}M parameters")
    logger.info(f"Backbone: ConvNextV2 {backbone_config.hidden_sizes} (huge)")
    logger.info(f"BiFPN: {bifpn_config.num_blocks} blocks, {bifpn_config.block.num_features} features")
    logger.info(f"Input size: {backbone_config.image_size}×{backbone_config.image_size}")

    return model




class PeakNetWorkload(ComputeWorkload):
    """
    PeakNet workload implementation for the high-performance inference pipeline.

    Implements the ComputeWorkload interface to integrate PeakNet with the
    double-buffered inference pipeline for maximum GPU utilization.
    """

    def __init__(self, yaml_path: str, weights_path: Optional[str] = None, 
                 mixed_precision_dtype: torch.dtype = torch.bfloat16):
        """
        Initialize PeakNet workload.

        Args:
            yaml_path: Path to YAML configuration file
            weights_path: Optional path to pre-trained weights  
            mixed_precision_dtype: Data type for mixed precision
        """
        self.yaml_path = yaml_path
        self.weights_path = weights_path
        self.mixed_precision_dtype = mixed_precision_dtype
        self.model = None

        # Load configuration to get input size
        config = OmegaConf.load(yaml_path)
        backbone_config = config.model.backbone.hf_config
        self.input_size = backbone_config.get("image_size", 1920)
        self.num_classes = config.model.seg_head.get("num_classes", 2)

        logger.info(f"PeakNetWorkload initialized for {self.input_size}×{self.input_size} images")

    def create_model(self, gpu_id: int, compile_options: dict) -> nn.Module:
        """Create and configure the PeakNet model."""
        # Load model
        model = load_peaknet_from_yaml(self.yaml_path, self.weights_path)

        # Move to GPU
        device = f'cuda:{gpu_id}'
        model = model.to(device)
        model.eval()

        # Note: torch.compile is handled by the pipeline base class
        # to avoid duplicate compilation

        self.model = model
        return model

    def get_input_buffer_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get GPU input buffer shape.

        Args:
            input_shape: Input tensor shape (C, H, W) - no batch dimension

        Returns:
            GPU input buffer shape for single-channel detector images
        """
        # PeakNet takes single-channel input: (1, H, W)
        C, H, W = input_shape
        return (1, H, W)  # Force single channel for detector images

    def get_output_buffer_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get CPU output buffer shape for segmentation results.

        Args:
            input_shape: Input tensor shape (C, H, W) - no batch dimension

        Returns:
            CPU output buffer shape for multi-class segmentation
        """
        # Output shape: (num_classes, H, W) for segmentation
        C, H, W = input_shape
        return (self.num_classes, H, W)

    def preprocess_batch(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Preprocess batch of tensors before GPU transfer.

        Args:
            batch: List of input tensors

        Returns:
            Preprocessed batch tensor
        """
        # Stack tensors into batch
        batch_tensor = torch.stack(batch, dim=0)

        # Convert to mixed precision dtype
        batch_tensor = batch_tensor.to(dtype=self.mixed_precision_dtype)

        # Normalize to [0, 1] if needed (assuming input is in [0, 255] or similar)
        if batch_tensor.max() > 2.0:  # Likely in [0, 255] range
            batch_tensor = batch_tensor / 255.0

        return batch_tensor

    def forward(self, gpu_tensor: torch.Tensor) -> torch.Tensor:
        """
        Execute PeakNet inference.

        Args:
            gpu_tensor: Input tensor on GPU

        Returns:
            Peak detection output tensor
        """
        with torch.no_grad():
            # Run inference
            output = self.model(gpu_tensor)

            # Apply softmax to get probabilities
            output = torch.softmax(output, dim=1)

        return output

    def postprocess_result(self, result: torch.Tensor) -> torch.Tensor:
        """
        Postprocess inference results.

        Args:
            result: Raw model output

        Returns:
            Processed results
        """
        # Convert probabilities to class predictions
        predictions = torch.argmax(result, dim=1, keepdim=True)

        # Convert to float32 for compatibility
        return predictions.float()


def main():
    """Main function for PeakNet inference."""
    parser = argparse.ArgumentParser(description='High-Performance PeakNet Inference')

    # Model configuration
    parser.add_argument('--yaml-path', type=str, default='experiments/yaml/peaknet-673m.yaml',
                       help='Path to YAML configuration file')
    parser.add_argument('--weights-path', type=str, default=None,
                       help='Path to pre-trained weights')

    # Hardware configuration  
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch-size', type=int, required=True,
                       help='Batch size for inference')

    # Performance options
    parser.add_argument('--compile-mode', type=str, default='none',
                       choices=['none', 'default', 'reduce-overhead', 'max-autotune'],
                       help='Torch compilation mode (none = disabled)')
    parser.add_argument('--warmup-samples', type=int, default=50,
                       help='Number of warmup samples for pipeline performance optimization (default: 50)')
    parser.add_argument('--warmup-batch-size', type=int, default=None,
                       help='Batch size for compilation warmup. '
                            'None (default): Enable CUDA graphs for 10-20%% speedup with reduce-overhead/max-autotune modes. '
                            '1: Memory-optimized compilation if you get OOM during compilation (disables CUDA graphs).')
    parser.add_argument('--mixed-precision', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Mixed precision data type')
    
    # Compilation caching options (PyTorch 2.0+)
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Custom directory for torch.compile cache artifacts (default: /tmp/torchinductor_<username>)')
    parser.add_argument('--cache-artifacts', type=str, default=None,
                       help='Path to save/load portable compilation cache artifacts. '
                            'Dramatically speeds up subsequent runs by avoiding recompilation. '
                            'Example: --cache-artifacts ./peaknet_cache.bin')
    parser.add_argument('--disable-mega-cache', action='store_true',
                       help='Disable end-to-end compilation caching (portable cache artifacts)')

    # Testing options
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of test samples for benchmarking')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--input-size', type=int, default=None,
                       help='Override input image size')

    args = parser.parse_args()

    logger.info("=== High-Performance PeakNet Inference ===")
    logger.info(f"Configuration: {args.yaml_path}")
    logger.info(f"GPU: {args.gpu_id}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info(f"Torch compile: {args.compile_mode}")
    if args.compile_mode != 'none':
        logger.info(f"Warmup samples: {args.warmup_samples}")

        # Log compilation strategy based on warmup_batch_size (simple and clear)
        if args.warmup_batch_size is None:
            logger.info(f"Compilation strategy: PERFORMANCE (CUDA graphs enabled)")
            if args.compile_mode in ['reduce-overhead', 'max-autotune']:
                logger.info(f"CUDA graph optimization: ENABLED (expect 10-20% speedup)")
                logger.info(f"Note: Warmup samples will be auto-increased to 1000+ for graph capture")
        else:
            logger.info(f"Compilation strategy: MEMORY (warmup_batch_size={args.warmup_batch_size})")
            logger.info(f"CUDA graph optimization: DISABLED (using dynamic compilation)")
            if args.compile_mode in ['reduce-overhead', 'max-autotune']:
                logger.info(f"Note: Dynamic compilation may limit CUDA graph benefits")

    # Determine CUDA graph preference: enabled by default, disabled only for memory optimization
    prefer_cuda_graphs = (args.warmup_batch_size is None)
    
    # Setup compilation caching
    enable_mega_cache = not args.disable_mega_cache
    if args.cache_artifacts and enable_mega_cache:
        logger.info(f"Compilation cache artifacts: {args.cache_artifacts}")
        if Path(args.cache_artifacts).exists():
            logger.info("✓ Cache file found - will attempt to load and skip warmup")
        else:
            logger.info("⚠ Cache file not found - will compile and save cache")
    
    if args.cache_dir:
        logger.info(f"Custom cache directory: {args.cache_dir}")
        
    if not enable_mega_cache:
        logger.info("Mega-cache disabled - will use local caching only")

    # Setup mixed precision dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16, 
        'bfloat16': torch.bfloat16
    }
    mixed_precision_dtype = dtype_map[args.mixed_precision]

    # Create workload
    workload = PeakNetWorkload(
        yaml_path=args.yaml_path,
        weights_path=args.weights_path,
        mixed_precision_dtype=mixed_precision_dtype
    )

    # Determine input shape and batch size
    input_size = args.input_size or workload.input_size
    input_shape = (1, input_size, input_size)  # Single channel

    batch_size = args.batch_size
    logger.info(f"Using batch size: {batch_size}")

    # Configure pipeline
    config = PipelineConfig(
        gpu_id=args.gpu_id,
        batch_size=batch_size,
        tensor_shape=input_shape,
        pin_memory=True,
        compile_model=(args.compile_mode != 'none'),
        compile_mode=args.compile_mode,
        warmup_samples=args.warmup_samples,
        warmup_batch_size=args.warmup_batch_size,
        prefer_cuda_graphs=prefer_cuda_graphs,
        memory_pool_size_mb=1024,
        # Compilation caching
        cache_dir=args.cache_dir,
        enable_mega_cache=enable_mega_cache,
        cache_artifacts_path=args.cache_artifacts
    )

    # Create pipeline
    pipeline = DoubleBufferedPipelineBase(config, workload)

    if args.benchmark:
        # Generate test data - pipeline handles warmup internally
        logger.info(f"Generating {args.num_samples} test samples...")
        test_data = generate_test_tensors(
            args.num_samples, 
            input_shape, 
            pin_memory=True,
            fill_pattern='random'
        )
        # Convert to desired dtype
        test_data = [tensor.to(dtype=mixed_precision_dtype) for tensor in test_data]

        logger.info("Running inference with automatic warmup...")
        start_time = time.time()

        # Single call - handles warmup automatically
        pipeline.run_inference_with_warmup(test_data, "peaknet_inference")

        end_time = time.time()
        total_time = end_time - start_time
        throughput = args.num_samples / total_time

        logger.info("=== Benchmark Results ===")
        logger.info(f"Test samples: {args.num_samples}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Throughput: {throughput:.2f} samples/s")
        logger.info(f"Latency per sample: {1000/throughput:.2f}ms")

    else:
        # Simple test with a few samples - pipeline handles warmup internally
        logger.info("Running simple test...")
        test_samples = min(10, batch_size * 2)

        # Generate test data
        test_data = generate_test_tensors(
            test_samples, 
            input_shape,
            pin_memory=True
        )
        # Convert to desired dtype
        test_data = [tensor.to(dtype=mixed_precision_dtype) for tensor in test_data]

        # Single call - handles warmup automatically
        pipeline.run_inference_with_warmup(test_data, "test")

        logger.info(f"✓ Processed {len(test_data)} samples successfully")


if __name__ == "__main__":
    main()
