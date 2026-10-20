#!/usr/bin/env python3
"""Run ThyroidXL inference on DICOM images or NIfTI volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from thyroidxl.inference import discover_inputs, predict_file, save_prediction_manifest
from thyroidxl.model import ThyroidXL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A DICOM/NIfTI file or directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ThyroidXL(encoder_weights=None)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.to(device).eval()
    inputs = discover_inputs(args.input)
    if not inputs:
        raise RuntimeError(f"No DICOM or NIfTI files found under {args.input}")
    predictions = [
        predict_file(model, path, args.output_dir, args.image_size, device, args.threshold)
        for path in inputs
    ]
    manifest = args.output_dir / "predictions.json"
    save_prediction_manifest(manifest, predictions)
    print(f"Saved {len(predictions)} prediction(s) to {manifest}")


if __name__ == "__main__":
    main()
