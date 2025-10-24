#!/usr/bin/env python3
"""
Script to downsample and/or normalize a .raw volumetric dataset
"""
import numpy as np
import argparse
import pathlib
import re
import sys

_DTYPE_ALIASES = {
    "float32": np.float32,
    "float64": np.float64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
}

def load_raw(filepath, shape, dtype=np.uint8):
    """Load raw binary file as numpy array"""
    data = np.fromfile(filepath, dtype=dtype,format="<f")
    return data.reshape(shape)

def downsample_volume(volume, factor):
    """
    Downsample volume by a given factor using average pooling
    factor=2 means half the size in each dimension (1/8 total volume)
    """
    factor = int(round(factor))

    if factor == 1:
        return volume

    # Calculate new shape
    new_shape = tuple(s // factor for s in volume.shape)

    # Reshape and average blocks
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

def normalize_volume(volume):
    """
    Normalize volume to float32 values in the 0-1 range
    """
    data = volume.astype(np.float32)

    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        raise ValueError("Input volume does not contain any finite values")
    if not finite_mask.all():
        raise ValueError("Input volume contains non-finite values")

    data_min = float(data.min())
    data_max = float(data.max())

    if data_max == data_min:
        normalized = np.zeros_like(data)
    else:
        scale = float(data_max - data_min)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Input data range is not finite")

        normalized = (data - data_min) / scale

        if not np.isfinite(normalized).all():
            raise ValueError("Normalization produced non-finite values")

        normalized = normalized.astype(np.float32)

    return normalized

def save_raw(filepath, data):
    """Save numpy array as raw binary file"""
    data.tofile(filepath)

def infer_dtype_from_name(path):
    """Infer dtype from filename"""
    tokens = re.split(r"[^0-9a-zA-Z]+", path.stem.lower())
    for token in tokens:
        if token in _DTYPE_ALIASES:
            return token
    return None

def main():
    parser = argparse.ArgumentParser(
        description='Downsample and/or normalize a .raw volumetric dataset'
    )
    parser.add_argument('input', type=str, help='Input .raw file path')
    parser.add_argument('output', type=str, help='Output .raw file path')
    parser.add_argument('--shape', type=int, nargs=3, required=True,
                        help='Input shape (e.g., 1024 1024 1080)')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=list(_DTYPE_ALIASES.keys()),
                        help='Data type of input raw file (will attempt to infer from filename if not provided)')
    parser.add_argument('--downsample', type=int, default=None,
                        help='Downsample factor (e.g., 2 for half size, 4 for quarter size)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize output to float32 range [0, 1]')

    args = parser.parse_args()

    # Determine input dtype
    if args.dtype:
        dtype = _DTYPE_ALIASES[args.dtype]
        dtype_name = args.dtype
    else:
        input_path = pathlib.Path(args.input)
        inferred = infer_dtype_from_name(input_path)
        if inferred:
            dtype = _DTYPE_ALIASES[inferred]
            dtype_name = inferred
            print(f"Inferred dtype '{inferred}' from input filename")
        else:
            dtype = np.float32
            dtype_name = 'float32'
            print(f"Using default dtype 'float32'")

    if not args.downsample and not args.normalize:
        print("Error: Must specify at least one of --downsample or --normalize")
        return 1

    print(f"Loading {args.input} with shape {args.shape} and dtype {dtype_name}...")
    volume = load_raw(args.input, tuple(args.shape), dtype)
    print(f"Original shape: {volume.shape}, size: {volume.nbytes / 1024**2:.2f} MB")

    # Downsample if requested
    if args.downsample:
        print(f"Downsampling by factor {args.downsample}...")
        volume = downsample_volume(volume, args.downsample)
        print(f"Downsampled shape: {volume.shape}, size: {volume.nbytes / 1024**2:.2f} MB")

    # Normalize if requested
    if args.normalize:
        print("Normalizing to [0, 1] range...")
        volume = normalize_volume(volume)
        print(f"Normalized to float32")

    print(f"Final shape: {volume.shape}, size: {volume.nbytes / 1024**2:.2f} MB")
    print(f"Saving to {args.output}...")
    save_raw(args.output, volume)

    print("Done!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
