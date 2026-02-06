import timm
import torch


def build_model(model_name: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model
