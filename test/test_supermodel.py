import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 

from fos import SuperModel, Trainer, NotebookMeter


def get_predictor():
    return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

def get_data(steps):
    return  [(torch.randn(16,10), torch.rand(16,1)) for i in range(steps)]


def test_supermodel_basic():
    predictor = get_predictor()
    loss      = F.mse_loss
    model = SuperModel(predictor, loss)
    
    assert model.step == 0
    
    data = get_data(1)[0]
    y = model(*data)
    assert y is not None
    assert model.step == 0
    
    y = model.validate(*data)
    assert y is not None
    assert model.step == 0
    
    y = model.predict(data[0])
    assert y is not None
    assert model.step == 0

    
def my_metric(y,t):
        assert y.shape == t.shape
        return 1.

def test_supermodel_metrics():

    metric_name = "test"
       
    predictor = get_predictor()
    loss      = F.mse_loss
    model = SuperModel(predictor, loss, metrics={metric_name: my_metric})
    assert model.step == 0
    
    data = get_data(1)[0]
    y = model.validate(*data)
    assert "val_" + metric_name in model.output
    assert "loss" not in model.output
    assert model.output["val_" + metric_name] == 1. 
    
    
    
def test_supermodel_train():
    metric_name = "test"
    
    predictor = get_predictor()
    loss      = F.mse_loss
    optim     = torch.optim.Adam(predictor.parameters())
    model = SuperModel(predictor, loss, metrics={metric_name: my_metric})
    
    assert model.step == 0
    
    x, t = get_data(1)[0]
    loss = model.learn(x, t, optim)
    assert loss is not None
    assert "loss" in model.output
    assert metric_name in model.output
    assert "val_loss" not in model.output
    assert model.step == 1
    
    loss2 = model.learn(x, t, optim)
    assert loss2 != loss
    assert model.step == 2