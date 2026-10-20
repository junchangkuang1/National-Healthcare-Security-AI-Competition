"""Shared runtime, logging, validation, and checkpoint utilities."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .objectives import batch_metrics, compute_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(
        self,
        *,
        epoch: int,
        step: int,
        phase: str,
        loss: Optional[float],
        learning_rate: Optional[float],
        data_source: str,
        checkpoint: Optional[str],
        pretrained_from: Optional[str],
    ) -> None:
        record = {
            "timestamp": utc_timestamp(), "epoch": int(epoch), "step": int(step),
            "phase": phase, "mode": "training", "loss": None if loss is None else float(loss),
            "lr": None if learning_rate is None else float(learning_rate),
            "data_source": data_source, "checkpoint": checkpoint, "pretrained_from": pretrained_from,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(line, flush=True)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def autocast_context(amp_dtype: str, device: torch.device):
    enabled = amp_dtype != "none" and device.type == "cuda" and torch.cuda.is_available()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_dice: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state = {
        "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "global_step": global_step, "best_dice": best_dice,
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "architecture": "UnetPlusPlus/timm-efficientnet-b5 ThyroidXL multitask",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, float]:
    model.eval()
    loss_sum, sample_count = 0.0, 0
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(amp_dtype, device):
            outputs = model(batch["image"])
            loss, _ = compute_loss(outputs, batch)
        count = batch["image"].shape[0]
        loss_sum += float(loss) * count
        sample_count += count
        for name, (value, metric_count) in batch_metrics(outputs, batch).items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + metric_count
    metrics = {name: totals[name] / max(1, counts[name]) for name in totals}
    metrics["loss"] = loss_sum / max(1, sample_count)
    return metrics
