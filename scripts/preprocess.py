#!/usr/bin/env python3
"""Validate raw annotations and create a patient-level dataset index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thyroidxl.data import ClinicalTable, IGNORE_INDEX, build_samples, save_samples, split_by_patient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--negative-frame-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bad-file-policy", choices=["skip", "error"], default="skip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spreadsheets = sorted(
        path for path in args.data_root.rglob("*.xlsx") if not path.name.startswith("~$")
    )
    if not spreadsheets:
        raise FileNotFoundError(f"No .xlsx annotation table found under {args.data_root}")
    clinical = ClinicalTable(spreadsheets)
    samples, bad_files = build_samples(
        args.data_root, clinical, args.negative_frame_ratio, args.seed, args.bad_file_policy
    )
    train, validation, test, split_ids = split_by_patient(samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_samples(args.output_dir / "samples.jsonl", samples)
    (args.output_dir / "patient_split_8_1_1.json").write_text(
        json.dumps(split_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "bad_nifti_files.json").write_text(
        json.dumps(bad_files, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "samples": {"all": len(samples), "train": len(train), "val": len(validation), "test": len(test)},
        "patients": {name: len(patient_ids) for name, patient_ids in split_ids.items()},
        "modalities": {
            "static": sum(sample.kind == "static" for sample in samples),
            "dynamic_frames": sum(sample.kind == "dynamic" for sample in samples),
        },
        "detection": {
            "positive": sum(sample.detection == 1 for sample in samples),
            "negative": sum(sample.detection == 0 for sample in samples),
        },
        "label_coverage": {
            field: sum(getattr(sample, field) != IGNORE_INDEX for sample in samples)
            for field in ("malignancy", "tirads", "extension", "location", "t_clinsig")
        },
        "corrupt_nifti_files": len(bad_files),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
