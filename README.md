# Transformer Image Processing (ViT)

Vision Transformer (ViT) image classification project with training, evaluation, and inference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset layout

This project expects an ImageNet-style folder:

```
data_dir/
	train/
		class_a/xxx.jpg
		class_b/yyy.jpg
	val/
		class_a/zzz.jpg
		class_b/uuu.jpg
```

For Tiny-ImageNet, set `num_classes: 200` in the config. For Mini-ImageNet, change to 100.

## Train

```bash
python main.py train --config configs/vit_tiny_imagenet.yaml
```

## Evaluate

```bash
python main.py eval --config configs/vit_tiny_imagenet.yaml --checkpoint checkpoints/best.pt
```

## Infer

```bash
python main.py infer --config configs/vit_tiny_imagenet.yaml --checkpoint checkpoints/best.pt --image /path/to/image.jpg
```

## Notes

- Logs are written to `runs/` for TensorBoard.
- Checkpoints are written to `checkpoints/`.
