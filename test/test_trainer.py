import torch
import torch.nn as nn
import torch.nn.functional as F
from fos import Supervisor, Trainer
from fos.meters import MemoryMeter
import os


def get_predictor():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1))


def get_data(steps):
    return [(torch.randn(16, 10), torch.rand(16, 1)) for i in range(steps)]


def test_trainer():
    predictor = get_predictor()
    loss = F.mse_loss
    optim = torch.optim.Adam(predictor.parameters())
    model = Supervisor(predictor, loss)
    meter = MemoryMeter()
    trainer = Trainer(model, optim, meter)

    data = get_data(100)
    trainer.run(data)
    assert trainer.epoch == 1
    
    trainer.run(data, epochs=10)
    assert trainer.epoch == 11
    
    valid_data = get_data(100)
    trainer.validate(valid_data)
    assert trainer.epoch == 11
    
    trainer.run(data, valid_data, epochs=5)
    assert trainer.epoch == 16
    
    
def test_trainer_state():
    predictor = get_predictor()
    loss = F.mse_loss
    optim = torch.optim.Adam(predictor.parameters())
    model = Supervisor(predictor, loss)
    meter = MemoryMeter()
    trainer = Trainer(model, optim, meter)

    state = trainer.state_dict()
    trainer = Trainer(model, optim, meter)
    trainer.load_state_dict(state)
    
    filename = "./tmp_file.dat"
    trainer.save(filename)
    trainer.load(filename)
    os.remove(filename)

def smart_metric(*args):
    return 1.
    
def test_trainer_metrics():
    predictor = get_predictor()
    loss = F.mse_loss
    optim = torch.optim.Adam(predictor.parameters())
    model = Supervisor(predictor, loss)
    meter = MemoryMeter()
    trainer = Trainer(model, optim, meter, metrics={"test":smart_metric})
 
    data = get_data(100)
    trainer.run(data, data)
    assert trainer.epoch == 1

