import torch
from torchvision.models import resnet18
from fos.utils import get_normalization, freeze, unfreeze
# pylint: disable=E1101


def test_freezer():
    model = resnet18()
    assert model.fc.weight.requires_grad
    freeze(model)
    assert not model.fc.weight.requires_grad
    unfreeze(model, "fc")
    assert model.fc.weight.requires_grad

def test_normalization():
    dataloader = torch.randn(1000, 100, 100)
    n = get_normalization(dataloader, 100)
    assert "mean" in n
    assert "std" in n
    assert len(n["mean"]) == 100
    
