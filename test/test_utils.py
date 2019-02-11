from unittest.mock import Mock

from fos.utils import *
from fos.utils.freezer import *
from fos.utils.normalization import *
import torch
import torch.nn.functional as F
from torchvision.models import resnet18



def test_freezer():
    model = resnet18()
    freezer = Freezer(model)
    freezer.summary()
    assert model.fc.weight.requires_grad == True
    freezer.freeze()
    assert model.fc.weight.requires_grad == False
    freezer.unfreeze("fc")
    assert model.fc.weight.requires_grad == True
    
def test_normalization():
    dataloader = torch.randn(1000,100,100)
    n = get_normalization(dataloader, 100)
    assert "mean" in n
    assert "std" in n
    assert len(n["mean"]) == 100