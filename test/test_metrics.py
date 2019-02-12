from unittest.mock import Mock

from fos import Supervisor
from fos.metrics import *
from fos.metrics.modelmetrics import *
from fos.metrics.confusion import *
import torch
from torchvision.models import resnet18


def test_accuracy():
    metric = BinaryAccuracy()
    y = torch.randn(100,10,10)
    result = metric(y, y>0.)
    assert result == 1.
    
    result = metric(y, y<0.)
    assert result == 0.
    
    metric = BinaryAccuracy(threshold=0.5, sigmoid=True)
    result = metric(y, y>0.)
    assert result == 1.
    
def test_tp():
    metric = ConfusionMetric(threshold=0.5, sigmoid=False)
    y = torch.FloatTensor([[0.1, 0.2, 0.8], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]])
    t = (y > 0.5).int()
    result = metric(y, t)
    assert "tp" in result
    assert result["tn"][0].item() == 2.  
    assert result["tp"][0].item() == 1.  
    
    
def test_learning_rates():
    model = resnet18()
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    lr = learning_rates(model, optim)
    assert lr == 0.05

def test_paramhistogram():
    predictor = resnet18()
    writer = Mock()
    loss = Mock()
    model = Supervisor(predictor, loss)
    metric = ParamHistogram(include_gradient=False, predictor_only=False)
    metric(model, None)
    assert writer.add_histogram.is_called()