import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from fos import Supervisor, Trainer
from fos.meters import NotebookMeter


def get_predictor():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1))


def get_data(steps):
    return [(torch.randn(16, 10), torch.rand(16, 1)) for i in range(steps)]


def test_supervisor_basic():
    predictor = get_predictor()
    loss = F.mse_loss
    model = Supervisor(predictor, loss)

    assert model.step == 0

    data = get_data(1)[0]
    y = model(*data)
    assert y is not None
    assert model.step == 0

    y = model.validate(*data)
    assert y is None
    assert model.step == 0

    y = model.predict(data[0])
    assert y is not None
    assert model.step == 0
    

def my_metric(y, t):
    assert y.shape == t.shape
    return 1.


class SmartMetric():
    def reset(self):
        pass
    
    def update(self, *args):
        pass
    
    def get(self):
        return "mymetric", 1.


def test_supervisor_metrics():

    metric_name = "test"

    predictor = get_predictor()
    loss = F.mse_loss
    model = Supervisor(predictor, loss, metrics=[SmartMetric()])
    assert model.step == 0

    data = get_data(1)[0]
    model.validate(*data)
    assert len(model.get_metrics()) == 2


def test_supervisor_train():
    metric_name = "test"

    predictor = get_predictor()
    loss = F.mse_loss
    optim = torch.optim.Adam(predictor.parameters())
    model = Supervisor(predictor, loss, metrics=[SmartMetric()])

    assert model.step == 0

    x, t = get_data(1)[0]
    model.learn(x, t)
    assert model.step == 1

    output2 = model.learn(x, t)
    assert model.step == 2

    
def test_supervisor_state():
    predictor = get_predictor()
    loss = F.mse_loss
    optim = torch.optim.Adam(predictor.parameters())
    model = Supervisor(predictor, loss)

    assert model.step == 0

    x, t = get_data(1)[0]
    model.learn(x, t)
    assert model.step == 1

    state = model.state_dict()
    model = Supervisor(predictor, loss)
    assert model.step == 0
    model.load_state_dict(state)
    assert model.step == 1
