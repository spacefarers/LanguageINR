#!/usr/bin/env python3
"""
Script to downsample a .raw volumetric dataset
"""
import numpy as np
import argparse

def load_raw(filepath, shape, dtype=np.uint8):
    """Load raw binary file as numpy array"""
    data = np.fromfile(filepath, dtype=dtype)
    return data.reshape(shape)

def downsample_volume(volume, factor):
    """
    Downsample volume by a given factor using average pooling
    factor=2 means half the size in each dimension (1/8 total volume)
    For non-integer factors, uses strided slicing
    """
    factor = int(round(factor))

    if factor == 1:
        return volume

    # Calculate new shape
    new_shape = tuple(s // factor for s in volume.shape)

    # Reshape and average blocks
    # This works by reshaping to include the pooling dimension
    temp_shape = (new_shape[0], factor,
                  new_shape[1], factor,
                  new_shape[2], factor)

    # Crop to make dimensions divisible by factor
    crop_shape = tuple(s // factor * factor for s in volume.shape)
    volume_cropped = volume[:crop_shape[0], :crop_shape[1], :crop_shape[2]]

    # Reshape and take mean
    reshaped = volume_cropped.reshape(temp_shape)
    downsampled = reshaped.mean(axis=(1, 3, 5))

    return downsampled

def save_raw(filepath, data):
    """Save numpy array as raw binary file"""
    data.tofile(filepath)

def main():
    parser = argparse.ArgumentParser(description='Downsample a .raw volumetric dataset')
    parser.add_argument('input', type=str, help='Input .raw file path')
    parser.add_argument('output', type=str, help='Output .raw file path')
    parser.add_argument('--shape', type=int, nargs=3, default=[1024, 1024, 1080],
                        help='Input shape (default: 1024 1024 1080)')
    parser.add_argument('--factor', type=int, default=2,
                        help='Downsample factor (default: 2 for 1/8 volume)')
    parser.add_argument('--dtype', type=str, default='uint8',
                        choices=['uint8', 'uint16', 'float32', 'float64'],
                        help='Data type of raw file (default: uint8)')

    args = parser.parse_args()

    # Map dtype string to numpy dtype
    dtype_map = {
        'uint8': np.uint8,
        'uint16': np.uint16,
        'float32': np.float32,
        'float64': np.float64
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.input} with shape {args.shape} and dtype {args.dtype}...")
    volume = load_raw(args.input, tuple(args.shape), dtype)

    print(f"Original shape: {volume.shape}, size: {volume.nbytes / 1024**2:.2f} MB")

    print(f"Downsampling by factor {args.factor}...")
    downsampled = downsample_volume(volume, args.factor)

    # Convert back to original dtype if needed
    if downsampled.dtype != dtype:
        downsampled = downsampled.astype(dtype)

    print(f"Downsampled shape: {downsampled.shape}, size: {downsampled.nbytes / 1024**2:.2f} MB")
    print(f"Size ratio: {volume.nbytes / downsampled.nbytes:.2f}x smaller")

    print(f"Saving to {args.output}...")
    save_raw(args.output, downsampled)

    print("Done!")

if __name__ == '__main__':
    main()
