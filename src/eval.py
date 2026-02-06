import yaml
import torch
import torch.nn as nn

from .data import get_data_loaders
from .models import build_model
from .utils import accuracy, get_device, load_checkpoint


def run_eval(config_path: str, checkpoint_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device(config.get("device", "auto"))

    _, val_loader, classes = get_data_loaders(
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

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    model.eval()

    losses = 0.0
    top1 = 0.0
    top5 = 0.0
    count = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            acc = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            losses += loss.item() * batch_size
            top1 += acc[1] * batch_size
            top5 += acc[5] * batch_size
            count += batch_size

    print(f"val loss {losses / count:.4f} top1 {top1 / count:.2f} top5 {top5 / count:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ViT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    args = parser.parse_args()

    run_eval(args.config, args.checkpoint)
