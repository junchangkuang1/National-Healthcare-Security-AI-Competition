# National Healthcare Security AI Competition

An anonymized, reproducible implementation of **ThyroidXL**, a multi-task thyroid-ultrasound model developed for the National Healthcare Security AI Competition.

The pipeline supports static DICOM images and dynamic NIfTI ultrasound volumes. A U-Net++ model with a `timm-efficientnet-b5` encoder jointly predicts lesion segmentation, lesion presence, malignancy, TI-RADS category, extrathyroidal extension, lesion location, and clinical T-stage significance.

> This repository contains source code only. No patient data, annotations, model weights, local paths, or identifying metadata are included.

## Repository layout

```text
.
├── configs/                 # Reproducible experiment configuration
├── docs/                    # Data and inference documentation
├── scripts/
│   ├── preprocess.py        # Validate data, build the index, and split patients
│   ├── train.py             # Train and validate the multi-task model
│   └── infer.py             # Run DICOM/NIfTI inference and export predictions
├── src/thyroidxl/
│   ├── data.py              # Clinical-table parsing, I/O, indexing, and Dataset
│   ├── model.py             # U-Net++ model and task heads
│   ├── objectives.py        # Losses and metrics
│   ├── training.py          # Training/validation utilities and checkpoints
│   └── inference.py         # Preprocessing and prediction export
└── tests/                   # Lightweight unit tests
```

## Installation

Python 3.10 or newer is recommended. Install a PyTorch build compatible with the local CUDA runtime first, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

The CUDA index URL above is only an example. Select the build recommended for the target GPU and driver from the official PyTorch installation guide.

## Data preparation

The expected dataset layout and supported spreadsheet fields are described in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md). Build a patient-level 8:1:1 split and validate all NIfTI inputs before training:

```bash
python scripts/preprocess.py \
  --data-root /path/to/training/annotation \
  --output-dir ./metadata
```

This creates `samples.jsonl`, `patient_split_8_1_1.json`, `summary.json`, and `bad_nifti_files.json`. The generated files are ignored by Git because they may contain patient identifiers and local paths.

## Training

```bash
python scripts/train.py \
  --index-file ./metadata/samples.jsonl \
  --split-file ./metadata/patient_split_8_1_1.json \
  --workspace ./workspace \
  --epochs 100 \
  --batch-size 6 \
  --image-size 384
```

The best and latest checkpoints are written to `workspace/checkpoints/`. Machine-readable JSONL records are written to `workspace/logs/training.log`.

## Inference

Run inference on one DICOM image, one NIfTI volume, or a directory containing either format:

```bash
python scripts/infer.py \
  --input /path/to/images \
  --checkpoint ./workspace/checkpoints/best_model.pth \
  --output-dir ./outputs
```

Predictions are summarized in `outputs/predictions.json`. DICOM masks are exported as PNG files and NIfTI masks as `.nii.gz` volumes. See [`docs/INFERENCE.md`](docs/INFERENCE.md) for the output schema and limitations.

## Reproducibility and privacy

- Random seeds are applied to Python, NumPy, and PyTorch.
- All train/validation/test partitions are created at the patient level.
- Corrupt compressed NIfTI files are detected before worker processes are launched.
- Local metadata and data-derived artifacts are excluded through `.gitignore`.
- This code does not upload data or predictions to any external service.

## Status

The repository is an anonymized competition code release. It does not include trained weights or numerical results; performance must be reproduced on the authorized dataset.

