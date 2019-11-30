from unittest.mock import Mock

from fos import Workout
from fos.metrics import *
import torch
from torchvision.models import resnet18


def test_accuracy():
    metric = BinaryAccuracy()
    y = torch.randn(100,10,10)
    value = metric(y, y>0.)
    assert value == 1.
    
    value = metric(y, y<0.)
    assert value == 0.
    
    
def test_tp():
    metric = ConfusionMetric(threshold=0.5, sigmoid=False)
    y = torch.FloatTensor([[0.1, 0.2, 0.8], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]])
    t = (y > 0.5).int()
    result = metric(y, t)
    assert len(result) == 4
    

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
    metric.update(model, None)
    assert writer.add_histogram.is_called()