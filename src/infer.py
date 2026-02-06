import json
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
import yaml

from .models import build_model
from .utils import get_device, load_checkpoint


def build_infer_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def run_infer(config_path: str, checkpoint_path: str, image_path: str, topk: int = 5) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device(config.get("device", "auto"))
    model = build_model(
        config["model_name"],
        int(config["num_classes"]),
        bool(config.get("pretrained", False)),
    )
    model.to(device)

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    classes = checkpoint.get("classes")

    model.eval()
    image = Image.open(image_path).convert("RGB")
    tf = build_infer_transform(int(config["image_size"]))
    tensor = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=topk)

    results: List[Tuple[str, float]] = []
    for v, i in zip(values.tolist(), indices.tolist()):
        label = classes[i] if classes else str(i)
        results.append((label, v))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Infer with ViT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    run_infer(args.config, args.checkpoint, args.image, args.topk)
