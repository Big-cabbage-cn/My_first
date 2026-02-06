import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_pref: str) -> torch.device:
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


@dataclass
class AverageMeter:
    name: str
    fmt: str = ".4f"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __str__(self) -> str:
        return f"{self.name} {self.val:{self.fmt}} (avg: {self.avg:{self.fmt}})"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[int, float]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0).item()
            res[k] = correct_k * 100.0 / target.size(0)
        return res


def save_checkpoint(state: Dict, checkpoint_dir: str, filename: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, device: torch.device) -> Dict:
    return torch.load(path, map_location=device)
