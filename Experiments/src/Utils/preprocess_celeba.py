#!/usr/bin/env python3
"""
Preprocess CelebA dataset to create CelebA32.pt tensor file.

This script loads raw CelebA images, applies transforms (resize, crop, grayscale),
and saves all images as a single tensor file for efficient training.

Usage:
    python preprocess_celeba.py \
        --raw-data-path /path/to/img_align_celeba \
        --output-path ../../Data/CelebA/
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the CelebADataset class from loader
import loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess CelebA dataset to create CelebA32.pt"
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        required=True,
        help="Path to the directory containing raw CelebA images (img_align_celeba)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../../Data/CelebA/",
        help="Path to save the output CelebA32.pt file (default: ../../Data/CelebA/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for loading images (default: 256)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=32,
        help="Target image size (default: 32)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Expand user path (for ~ on HPC)
    raw_data_path = os.path.expanduser(args.raw_data_path)
    output_path = os.path.expanduser(args.output_path)
    
    # Validate input path
    if not os.path.isdir(raw_data_path):
        print(f"Error: Raw data path does not exist: {raw_data_path}")
        sys.exit(1)
    
    # Check that list_attr_celeba.txt exists one directory above
    attr_file = os.path.join(os.path.dirname(raw_data_path), "list_attr_celeba.txt")
    if not os.path.isfile(attr_file):
        print(f"Warning: Attribute file not found at: {attr_file}")
        print("CelebADataset requires this file, but we'll try to proceed anyway.")
    
    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)
    
    # Define transforms: ToTensor, Resize(32), CenterCrop(32), Grayscale(1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.Grayscale(1),
    ])
    
    print(f"Loading CelebA dataset from: {raw_data_path}")
    print(f"Target image size: {args.size}x{args.size}")
    
    # Create dataset
    dataset = loader.CelebADataset(raw_data_path, transform=transform)
    print(f"Found {len(dataset)} images")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Collect all images
    all_images = []
    print("Processing images...")
    for batch in tqdm(dataloader, desc="Loading batches"):
        all_images.append(batch)
    
    # Concatenate into single tensor [N, 1, 32, 32]
    full_tensor = torch.cat(all_images, dim=0)
    print(f"Final tensor shape: {full_tensor.shape}")
    print(f"Tensor dtype: {full_tensor.dtype}")
    print(f"Value range: [{full_tensor.min():.4f}, {full_tensor.max():.4f}]")
    
    # Save tensor
    output_file = os.path.join(output_path, f"CelebA{args.size}.pt")
    print(f"Saving to: {output_file}")
    torch.save(full_tensor, output_file)
    
    # Print file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
