#!/usr/bin/env python3
"""Transformer image classification entry point."""

import argparse

from src.eval import run_eval
from src.infer import run_infer
from src.train import run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ViT image classification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument("--config", required=True, help="Path to YAML config")

    eval_p = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    eval_p.add_argument("--config", required=True, help="Path to YAML config")
    eval_p.add_argument("--checkpoint", required=True, help="Path to checkpoint")

    infer_p = subparsers.add_parser("infer", help="Run inference on one image")
    infer_p.add_argument("--config", required=True, help="Path to YAML config")
    infer_p.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    infer_p.add_argument("--image", required=True, help="Path to image")
    infer_p.add_argument("--topk", type=int, default=5)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args.config)
    elif args.command == "eval":
        run_eval(args.config, args.checkpoint)
    elif args.command == "infer":
        run_infer(args.config, args.checkpoint, args.image, args.topk)


if __name__ == "__main__":
    main()
