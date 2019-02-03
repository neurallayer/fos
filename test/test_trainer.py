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

    
def test_trainer():
    predictor = get_predictor()
    loss      = F.mse_loss
    model = SuperModel(predictor, loss)
    
    assert model.step == 0
    
    data = get_data(1)[0]
    y = model(*data)
    y = model.validate(*data)
    y = model.predict(data[0])