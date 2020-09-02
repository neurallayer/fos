# pylint: disable=E1101, C0116, C0114

from random import randint
import torch
import torch.nn as nn
from torchvision.models import resnet18
from fos.utils import get_normalization, freeze, unfreeze, print_params, init_random


def get_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1))


def test_random():
    init_random()
    a = randint(0, 1000)
    init_random()
    b = randint(0, 1000)
    assert a == b
    init_random(42, cudnn=True)


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
