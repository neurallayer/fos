import torch
from fos.metrics import BinaryAccuracy, ConfusionMetric
# pylint: disable=E1101


def test_accuracy():
    metric = BinaryAccuracy()
    y = torch.randn(100, 10, 10)
    value = metric(y, y > 0.)
    assert value == 1.0  
    value = metric(y, y < 0.)
    assert value == 0.0

def test_tp():
    metric = ConfusionMetric(threshold=0.5, sigmoid=False)
    y = torch.FloatTensor([[0.1, 0.2, 0.8], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]])
    t = (y > 0.5).int()
    result = metric(y, t)
    assert len(result) == 4
