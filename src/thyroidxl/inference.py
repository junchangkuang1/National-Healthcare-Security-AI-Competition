"""File discovery, preprocessing, and prediction export for inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import nibabel as nib
import numpy as np
import torch

from .data import image_to_tensor, is_mask_path, load_nifti_array, read_dicom_image, read_nifti_slice

CLASSIFICATION_HEADS = ("malignancy", "tirads", "extension", "location", "t_clinsig")


def discover_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    files = [*path.rglob("*.dcm"), *path.rglob("*.nii"), *path.rglob("*.nii.gz")]
    return sorted(
        set(item for item in files if item.is_file() and not is_mask_path(item))
    )


def _frame_predictions(
    model: torch.nn.Module,
    images: Iterable[np.ndarray],
    image_size: int,
    device: torch.device,
    threshold: float,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    masks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for frame_index, image in enumerate(images):
        tensor = image_to_tensor(image, image_size).unsqueeze(0).to(device)
        outputs = model(tensor)
        probability = torch.sigmoid(outputs["segmentation"])[0, 0].float().cpu().numpy()
        probability = cv2.resize(probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        masks.append((probability >= threshold).astype(np.uint8))
        record: dict[str, Any] = {
            "frame_index": frame_index,
            "detection_probability": float(torch.sigmoid(outputs["detection"])[0].cpu()),
        }
        for name in CLASSIFICATION_HEADS:
            probabilities = torch.softmax(outputs[name][0], dim=0).float().cpu().tolist()
            predicted_class = int(np.argmax(probabilities))
            if name in {"tirads", "extension", "location"}:
                predicted_class += 1
            record[name] = {"predicted_class": predicted_class, "probabilities": probabilities}
        records.append(record)
    return masks, records


@torch.inference_mode()
def predict_file(
    model: torch.nn.Module,
    input_path: Path,
    output_dir: Path,
    image_size: int,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lower_name = input_path.name.lower()
    if lower_name.endswith(".dcm"):
        image = read_dicom_image(str(input_path))
        masks, frames = _frame_predictions(model, [image], image_size, device, threshold)
        output_path = output_dir / f"{input_path.stem}_mask.png"
        cv2.imwrite(str(output_path), masks[0] * 255)
        kind = "dicom"
    elif lower_name.endswith(".nii") or lower_name.endswith(".nii.gz"):
        source = nib.load(str(input_path))
        array = np.squeeze(load_nifti_array(str(input_path)))
        frame_count = int(array.shape[2]) if array.ndim == 3 else 1
        images = [read_nifti_slice(str(input_path), frame) for frame in range(frame_count)]
        masks, frames = _frame_predictions(model, images, image_size, device, threshold)
        if frame_count == 1:
            volume = masks[0].T
        else:
            volume = np.stack([mask.T for mask in masks], axis=2)
        stem = input_path.name[:-7] if lower_name.endswith(".nii.gz") else input_path.stem
        output_path = output_dir / f"{stem}_mask.nii.gz"
        nib.save(nib.Nifti1Image(volume.astype(np.uint8), source.affine, source.header), str(output_path))
        kind = "nifti"
    else:
        raise ValueError(f"Unsupported input format: {input_path}")

    aggregate: dict[str, Any] = {
        "detection_probability": float(np.mean([frame["detection_probability"] for frame in frames])),
    }
    for name in CLASSIFICATION_HEADS:
        mean_probability = np.mean([frame[name]["probabilities"] for frame in frames], axis=0)
        predicted_class = int(np.argmax(mean_probability))
        if name in {"tirads", "extension", "location"}:
            predicted_class += 1
        aggregate[name] = {
            "predicted_class": predicted_class,
            "probabilities": mean_probability.tolist(),
        }
    return {
        "input": str(input_path), "input_type": kind, "mask": str(output_path),
        "frame_count": len(frames), "aggregate": aggregate, "frames": frames,
    }


def save_prediction_manifest(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"predictions": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")
