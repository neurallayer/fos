# pylint: disable=E1101, C0116, C0114

import torch
from torchvision.models import resnet18
from fos.utils import get_normalization, freeze, unfreeze, print_params
import torch.nn as nn

def get_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1))


def test_freezer():
    model = resnet18()
    assert model.fc.weight.requires_grad
    freeze(model)
    assert not model.fc.weight.requires_grad
    unfreeze(model, "fc")
    assert model.fc.weight.requires_grad

def test_normalization():
    dataloader = torch.randn(1000, 100, 100)
    norm = get_normalization(dataloader, 100)
    assert "mean" in norm
    assert "std" in norm
    assert len(norm["mean"]) == 100


def test_print():
    model = get_model()
    print_params(model)