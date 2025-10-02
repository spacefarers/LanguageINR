#!/usr/bin/env python3
"""Normalize a raw volume file to float32 values in the 0-1 range."""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Optional

import numpy as np

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


def _infer_dtype_from_name(path: pathlib.Path) -> Optional[str]:
    tokens = re.split(r"[^0-9a-zA-Z]+", path.stem.lower())
    for token in tokens:
        if token in _DTYPE_ALIASES:
            return token
    return None


def _resolve_dtype(input_path: pathlib.Path, user_dtype: Optional[str]) -> np.dtype:
    if user_dtype is not None:
        key = user_dtype.lower()
        try:
            return _DTYPE_ALIASES[key]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported dtype '{user_dtype}'. Supported: {', '.join(sorted(_DTYPE_ALIASES))}"
            ) from exc

    inferred = _infer_dtype_from_name(input_path)
    if inferred is not None:
        print(f"Inferred dtype '{inferred}' from input filename", file=sys.stderr)
        return _DTYPE_ALIASES[inferred]

    return np.float32


def normalize_volume(
    input_path: pathlib.Path,
    dtype: Optional[str],
    output_path: Optional[pathlib.Path],
) -> pathlib.Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    resolved_dtype = _resolve_dtype(input_path, dtype)
    data = np.fromfile(input_path, dtype=resolved_dtype)
    if data.size == 0:
        raise ValueError(f"Input file appears to be empty: {input_path}")

    data = data.astype(np.float32)
    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        raise ValueError(
            "Input volume does not contain any finite values after conversion; check the dtype argument."
        )
    if not finite_mask.all():
        raise ValueError(
            "Input volume contains non-finite values after conversion; check the dtype argument."
        )

    data_min = float(data.min())
    data_max = float(data.max())

    if data_max == data_min:
        normalized = np.zeros_like(data)
    else:
        scale = float(data_max - data_min)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "Input data range is not finite after conversion; check the dtype argument."
            )

        with np.errstate(divide="raise", invalid="raise", over="raise"):
            try:
                normalized = np.subtract(
                    data, data_min, dtype=np.float64, casting="unsafe"
                )
                normalized = np.divide(normalized, scale, dtype=np.float64)
            except FloatingPointError as exc:
                raise ValueError(
                    "Numerical overflow detected during normalization; the input dtype is likely incorrect."
                ) from exc

        if not np.isfinite(normalized).all():
            raise ValueError(
                "Normalization produced non-finite values; the input dtype is likely incorrect."
            )

        normalized = normalized.astype(np.float32)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_normalized_float32.raw")

    normalized.astype(np.float32).tofile(output_path)
    return output_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a raw volume file to float32 values in the 0-1 range.",
    )
    parser.add_argument("input", type=pathlib.Path, help="Path to the raw volume file")
    parser.add_argument(
        "--dtype",
        help=(
            "Input data type. Supported: "
            + ", ".join(sorted(_DTYPE_ALIASES))
            + ". Defaults to float32."
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Optional path for the normalized file. Defaults to <name>_normalized_float32.raw",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        output_path = normalize_volume(args.input, args.dtype, args.output)
    except Exception as exc:  # noqa: BLE001 - surface to CLI
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote normalized volume to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
