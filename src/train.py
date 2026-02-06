import os
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml

from .data import get_data_loaders
from .models import build_model
from .utils import AverageMeter, accuracy, get_device, save_checkpoint, set_seed


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    losses = AverageMeter("loss")
    top1_meter = AverageMeter("top1")
    top5_meter = AverageMeter("top5")

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1_meter.update(acc[1], images.size(0))
        top5_meter.update(acc[5], images.size(0))

    return {"loss": losses.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


def evaluate(model: torch.nn.Module, loader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses = AverageMeter("loss")
    top1_meter = AverageMeter("top1")
    top5_meter = AverageMeter("top5")

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            acc = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1_meter.update(acc[1], images.size(0))
            top5_meter.update(acc[5], images.size(0))

    return {"loss": losses.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


def run_train(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config["seed"]))
    device = get_device(config.get("device", "auto"))

    train_loader, val_loader, classes = get_data_loaders(
        config["data_dir"],
        int(config["image_size"]),
        int(config["batch_size"]),
        int(config["num_workers"]),
    )

    model = build_model(
        config["model_name"],
        int(config["num_classes"]),
        bool(config.get("pretrained", False)),
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))

    use_amp = bool(config.get("amp", True)) and device.type != "cpu"
    scaler = GradScaler(enabled=use_amp)

    run_name = config.get("project_name", "vit")
    log_dir = os.path.join(config.get("log_dir", "runs"), run_name)
    writer = SummaryWriter(log_dir=log_dir)

    best_top1 = 0.0
    for epoch in range(1, int(config["epochs"]) + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/top1", train_metrics["top1"], epoch)
        writer.add_scalar("train/top5", train_metrics["top5"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/top1", val_metrics["top1"], epoch)
        writer.add_scalar("val/top5", val_metrics["top5"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"train loss {train_metrics['loss']:.4f} top1 {train_metrics['top1']:.2f} | "
            f"val loss {val_metrics['loss']:.4f} top1 {val_metrics['top1']:.2f} | "
            f"{elapsed:.1f}s"
        )

        is_best = val_metrics["top1"] > best_top1
        if is_best:
            best_top1 = val_metrics["top1"]

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_top1": best_top1,
            "classes": classes,
            "config": config,
        }
        save_checkpoint(state, config.get("checkpoint_dir", "checkpoints"), "last.pt")
        if is_best:
            save_checkpoint(state, config.get("checkpoint_dir", "checkpoints"), "best.pt")

    writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ViT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    run_train(args.config)
