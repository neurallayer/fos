# pylint: disable=E1101, C0116, C0114
import torch
from fos.metrics import BinaryAccuracy, ConfusionMetric

def test_accuracy():
    metric = BinaryAccuracy()
    y_pred = torch.randn(100, 10, 10)
    value = metric(y_pred, y_pred > 0.)
    assert value == 1.0
    value = metric(y_pred, y_pred < 0.)
    assert value == 0.0

def test_tp():
    metric = ConfusionMetric(threshold=0.5, sigmoid=False)
    y_pred = torch.FloatTensor([[0.1, 0.2, 0.8], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]])
    y_true = (y_pred > 0.5).int()
    result = metric(y_pred, y_true)
    assert len(result) == 4
