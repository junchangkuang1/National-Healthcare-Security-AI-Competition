#!/usr/bin/env python3
"""Train and validate the ThyroidXL multi-task network."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from thyroidxl.data import ThyroidDataset, load_samples, select_split
from thyroidxl.model import ThyroidXL
from thyroidxl.objectives import batch_metrics, compute_loss
from thyroidxl.training import (
    JsonlLogger,
    autocast_context,
    move_batch,
    save_checkpoint,
    set_seed,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--encoder-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 128 or args.grad_accum < 1:
        raise ValueError("epochs, batch-size, and grad-accum must be positive; image-size must be >= 128")
    set_seed(args.seed)
    logs_dir = args.workspace / "logs"
    checkpoints_dir = args.workspace / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "training.log"
    if args.resume is None and log_path.exists():
        archive = logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.replace(archive)
    logger = JsonlLogger(log_path)

    samples = load_samples(args.index_file)
    split_ids = json.loads(args.split_file.read_text(encoding="utf-8"))
    train_samples = select_split(samples, split_ids, "train")
    validation_samples = select_split(samples, split_ids, "val")
    test_samples = select_split(samples, split_ids, "test")
    if not train_samples or not validation_samples:
        raise RuntimeError("The patient-level split contains an empty train or validation partition")

    train_dataset = ThyroidDataset(train_samples, args.image_size, training=True)
    validation_dataset = ThyroidDataset(validation_samples, args.image_size, training=False)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=len(train_dataset) >= args.batch_size,
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    encoder_weights: Optional[str] = None if args.encoder_weights == "none" else args.encoder_weights
    pretrained_from: Optional[str] = None
    try:
        model = ThyroidXL(encoder_weights=encoder_weights)
        pretrained_from = "official/imagenet/timm-efficientnet-b5" if encoder_weights else None
    except Exception as exc:
        if encoder_weights is None:
            raise
        print(f"[warning] ImageNet weights unavailable ({exc}); using random initialization.")
        model = ThyroidXL(encoder_weights=None)
    model.to(device)
    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.02
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16" and device.type == "cuda")
    start_epoch, global_step, best_dice = 1, 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_dice = float(checkpoint.get("best_dice", -1.0))
        pretrained_from = str(args.resume)

    parameter_count = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(
        f"[device] {device} | image={args.image_size} | batch={args.batch_size} "
        f"| AMP={args.amp_dtype} | parameters={parameter_count:.2f}M"
    )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss, running_batches = 0.0, 0
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with autocast_context(args.amp_dtype, device):
                outputs = model(batch["image"])
                loss, loss_parts = compute_loss(outputs, batch)
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            running_loss += float(loss.detach())
            running_batches += 1
            if global_step % args.log_every == 0 or batch_index == len(train_loader):
                logger.write(
                    epoch=epoch, step=global_step, phase="train",
                    loss=running_loss / max(1, running_batches),
                    learning_rate=optimizer.param_groups[0]["lr"],
                    data_source="competition/train", checkpoint=None,
                    pretrained_from=pretrained_from,
                )
                metrics = batch_metrics(outputs, batch)
                print(
                    f"[train] epoch={epoch} step={global_step} "
                    f"dice={metrics['dice'][0] / metrics['dice'][1]:.4f} "
                    f"iou={metrics['iou'][0] / metrics['iou'][1]:.4f} "
                    f"seg_bce={loss_parts['seg_bce']:.4f} dice_loss={loss_parts['seg_dice']:.4f}"
                )
                running_loss, running_batches = 0.0, 0

        metrics = validate(model, validation_loader, device, args.amp_dtype)
        scheduler.step()
        improved = metrics.get("dice", 0.0) > best_dice
        if improved:
            best_dice = metrics["dice"]
            save_checkpoint(
                checkpoints_dir / "best_model.pth", model, optimizer, scheduler, scaler,
                epoch, global_step, best_dice, args,
            )
        save_checkpoint(
            checkpoints_dir / "last_model.pth", model, optimizer, scheduler, scaler,
            epoch, global_step, best_dice, args,
        )
        checkpoint_name = "checkpoints/best_model.pth" if improved else "checkpoints/last_model.pth"
        logger.write(
            epoch=epoch, step=global_step, phase="val", loss=metrics["loss"],
            learning_rate=optimizer.param_groups[0]["lr"], data_source="competition/val",
            checkpoint=checkpoint_name, pretrained_from=pretrained_from,
        )
        metric_text = " ".join(f"{name}={value:.4f}" for name, value in sorted(metrics.items()))
        print(
            f"[val] epoch={epoch} {metric_text} best_dice={best_dice:.4f} "
            f"time={time.perf_counter() - epoch_start:.1f}s"
        )

    print(f"Training complete. Best checkpoint: {checkpoints_dir / 'best_model.pth'}")
    print(f"Held-out test patients remain unevaluated: {len(test_samples)} samples")


if __name__ == "__main__":
    main()
