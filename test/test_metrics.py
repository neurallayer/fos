from fos.metrics import *
from fos.metrics.modelmetrics import *
from fos.metrics.confusion import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18



def test_tp():
    metric = ConfusionMetric(threshold=0.5, sigmoid=False)
    y = torch.FloatTensor([[0.1, 0.2, 0.8], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]])
    t = (y > 0.5).int()
    result = metric(y, t)
    assert "tp" in result
    assert result["tn"][0].item() == 2.  
    assert result["tp"][0].item() == 1.  
    
    
def test_learning_rates():
    pass
