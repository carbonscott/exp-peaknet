import argparse
import numpy as np
import zarr
import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from peaknet.tensor_transforms import (
    Pad,
    PolarCenterCrop,
    MergeBatchPatchDims,
    NoTransform,
    InstanceNorm,
)
from peaknet_inference import PeakNetInference

class ZarrDataset(Dataset):
    def __init__(self, zarr_file, transforms=None):
        self.data = zarr.open(zarr_file, mode='r')['images']
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image[None,None])
        if self.transforms is not None:
            for enum_idx, trans in enumerate(self.transforms):
                image = trans(image)
        return image[0]

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def create_zarr_store(output_dir, input_name, partition):
    filepath = os.path.join(output_dir, f"{input_name}.p{partition:d}.zarr")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return zarr.open(filepath, mode='w'), filepath

def create_zarr_datasets(store, total_size, input_shape, feature_shape):
    store.create_dataset('inputs', shape=(total_size, *input_shape), chunks=(1, *input_shape), dtype=np.float32)
    store.create_dataset('features', shape=(total_size, *feature_shape), chunks=(1, *feature_shape), dtype=np.float32)

def setup_transforms(H_pad, W_pad, Hv, Wv, sigma, num_crop, uses_pad, uses_polar_center_crop):
    transforms = [
        Pad(H_pad, W_pad) if uses_pad else NoTransform(),
        PolarCenterCrop(Hv=Hv, Wv=Wv, sigma=sigma, num_crop=num_crop) if uses_polar_center_crop else NoTransform(),
        MergeBatchPatchDims() if uses_polar_center_crop else NoTransform(),
    ]

    return transforms

def process_zarr(input_file, batch_size, partition_size, output_dir, inferencer, transform_params, device, logger):
    logger.info(f"Starting processing for input file: {input_file}")

    # Set up dataset and dataloader
    transforms = setup_transforms(**transform_params)
    instance_norm = InstanceNorm()

    dataset = ZarrDataset(input_file, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Process data
    store           = None
    partition       = 0
    total_processed = 0
    zarr_path       = None
    input_buffer    = []
    features_buffer = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)

            # Norm input
            norm_batch = instance_norm(batch)

            # Perform inference
            features = inferencer.predict(norm_batch, returns_features=True)

            batch    = batch.cpu().numpy()
            features = features.cpu().numpy()

            # Write data
            for input, feature in zip(batch, features):
                input_buffer.append(input)
                features_buffer.append(feature)

                total_processed += 1
                if total_processed == partition_size:
                    # Create output Zarr store
                    if store is None:
                        store, zarr_path = create_zarr_store(output_dir, os.path.splitext(os.path.basename(input_file))[0], partition)
                        logger.info(f"Creating zarr file {zarr_path}")

                    # Create datasets if not exist
                    if 'inputs' not in store:
                        create_zarr_datasets(store, len(input_buffer), input_buffer[0].shape, features_buffer[0].shape)

                    for i, data in enumerate(input_buffer):
                        store.inputs[i] = data
                    for i, data in enumerate(features_buffer):
                        store.features[i] = data

                    logger.info(f"{total_processed} data points are saved in {zarr_path}")

                    store = None
                    partition += 1
                    total_processed = 0
                    zarr_path = None
                    input_buffer    = []
                    features_buffer = []

        if input_buffer:
            # Create output Zarr store
            if store is None:
                store, zarr_path = create_zarr_store(output_dir, os.path.splitext(os.path.basename(input_file))[0], partition)
                logger.info(f"Creating zarr file {zarr_path}")

            # Create datasets if not exist
            if 'inputs' not in store:
                create_zarr_datasets(store, len(input_buffer), input_buffer[0].shape, features_buffer[0].shape)

            for i, data in enumerate(input_buffer):
                store.inputs[i] = data
            for i, data in enumerate(features_buffer):
                store.features[i] = data

            logger.info(f"{total_processed} final data points are saved in {zarr_path}")

    return True

def main():
    parser = argparse.ArgumentParser(description="Process multiple Zarr data files, perform PeakNet inference, and save results as Zarr files")
    parser.add_argument("--input-files", nargs='+', required=True, help="List of input Zarr file paths")
    parser.add_argument("--config-path", required=True, help="Path to PeakNet config file")
    parser.add_argument("--weights-path", required=True, help="Path to PeakNet weights file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--partition-size", type=int, default=1000, help="Number of images per partition")
    parser.add_argument("--output-dir", default="./output", help="Directory to save output Zarr files (default: ./output)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")

    # Add transform parameters
    parser.add_argument("--H-pad", type=int, default=0, help="Padding height")
    parser.add_argument("--W-pad", type=int, default=0, help="Padding width")
    parser.add_argument("--Hv", type=int, required=True, help="Vertical size for PolarCenterCrop")
    parser.add_argument("--Wv", type=int, required=True, help="Horizontal size for PolarCenterCrop")
    parser.add_argument("--sigma", type=float, required=True, help="Sigma for PolarCenterCrop")
    parser.add_argument("--num-crop", type=int, default=1, help="Number of crops for PolarCenterCrop")
    parser.add_argument("--uses-pad", action="store_true", help="Use padding transform")
    parser.add_argument("--uses-polar-center-crop", action="store_true", help="Use PolarCenterCrop transform")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(getattr(logging, args.log_level))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare transform parameters
    transform_params = {
        "H_pad"                 : args.H_pad,
        "W_pad"                 : args.W_pad,
        "Hv"                    : args.Hv,
        "Wv"                    : args.Wv,
        "sigma"                 : args.sigma,
        "num_crop"              : args.num_crop,
        "uses_pad"              : args.uses_pad,
        "uses_polar_center_crop": args.uses_polar_center_crop,
    }

    logger.info(f"Input files: {args.input_files}")
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    logger.info(f"Config path: {args.config_path}")
    logger.info(f"Weights path: {args.weights_path}")
    logger.info(f"Device: {device}")

    # Initialize PeakNet model once
    inferencer = PeakNetInference(args.config_path, args.weights_path)
    inferencer.model.to(device)
    inferencer.model.eval()
    logger.info("PeakNet inference model initialized")

    for input_file in args.input_files:
        logger.info(f"Processing file: {input_file}")
        success = process_zarr(input_file, args.batch_size, args.partition_size, args.output_dir, inferencer, transform_params, device, logger)

        if success:
            logger.info(f"Processing completed successfully for {input_file}.")
        else:
            logger.error(f"Processing failed for {input_file}.")

    logger.info("All files processed.")

if __name__ == "__main__":
    main()
