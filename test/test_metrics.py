from fos.metrics import * 
from fos.metrics.modelmetrics import * 
from fos.metrics.precision import * 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 


def get_predictor():
    return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

def test_tp():
    metric = TPMetric()
    y = torch.FloatTensor([[0.1,0.2,0.8], [0.4,0.5,0.6], [0.6,0.7,0.8]])
    tp = metric(y, y.clone())
     
    
    
def test_learning_rates():
    pass
