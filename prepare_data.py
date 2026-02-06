#!/usr/bin/env python3
"""Reorganize Tiny-ImageNet dataset to standard ImageFolder format."""

import os
import shutil
from pathlib import Path


def prepare_tiny_imagenet(source_dir, target_dir):
    """
    Reorganize Tiny-ImageNet from:
      train/n01443537/images/*.JPEG
      val/images/*.JPEG (with val_annotations.txt)

    To standard format:
      train/n01443537/*.JPEG
      val/n01443537/*.JPEG
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print()

    # Create target directories
    target_train = target_dir / "train"
    target_val = target_dir / "val"
    target_train.mkdir(parents=True, exist_ok=True)
    target_val.mkdir(parents=True, exist_ok=True)

    # ========== Reorganize Training Data ==========
    print("Reorganizing training data...")
    source_train = source_dir / "train"

    if not source_train.exists():
        print(f"Error: {source_train} does not exist!")
        return

    class_dirs = sorted([d for d in source_train.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes in training data")

    for class_dir in class_dirs:
        class_id = class_dir.name
        images_dir = class_dir / "images"

        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue

        # Create target class directory
        target_class_dir = target_train / class_id
        target_class_dir.mkdir(exist_ok=True)

        # Copy images
        image_files = list(images_dir.glob("*.JPEG"))
        for img in image_files:
            target_img = target_class_dir / img.name
            if not target_img.exists():
                shutil.copy2(img, target_img)

        print(f"  {class_id}: {len(image_files)} images")

    print()

    # ========== Reorganize Validation Data ==========
    print("Reorganizing validation data...")
    source_val = source_dir / "val"
    val_annotations = source_val / "val_annotations.txt"
    val_images = source_val / "images"

    if not val_annotations.exists():
        print(f"Error: {val_annotations} does not exist!")
        return

    # Read annotations
    img_to_class = {}
    with open(val_annotations) as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name = parts[0]
            class_id = parts[1]
            img_to_class[img_name] = class_id

    print(f"Found {len(img_to_class)} validation images")

    # Create class directories and copy images
    class_counts = {}
    for img_name, class_id in img_to_class.items():
        source_img = val_images / img_name

        if not source_img.exists():
            continue

        # Create target class directory
        target_class_dir = target_val / class_id
        target_class_dir.mkdir(exist_ok=True)

        # Copy image
        target_img = target_class_dir / img_name
        if not target_img.exists():
            shutil.copy2(source_img, target_img)

        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    print(f"  Organized into {len(class_counts)} classes")
    print(f"  Average {sum(class_counts.values()) / len(class_counts):.0f} images per class")
    print()

    print("=" * 50)
    print("✓ Data preparation complete!")
    print(f"✓ Training data: {target_train}")
    print(f"✓ Validation data: {target_val}")
    print("=" * 50)


if __name__ == "__main__":
    # Source: original downloaded Tiny-ImageNet
    source = "/Users/diaobaocheng/Downloads/tiny-imagenet-200"

    # Target: organized data in project directory
    target = "/Users/diaobaocheng/my-first-project/data/tiny-imagenet-200"

    prepare_tiny_imagenet(source, target)
