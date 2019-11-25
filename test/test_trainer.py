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
    
    
def test_trainer_predict():
    
    predictor = get_predictor()
    loss = F.mse_loss
    optim = torch.optim.Adam(predictor.parameters())
    model = Supervisor(predictor, loss)
    meter = MemoryMeter()
    trainer = Trainer(model, optim, meter)
    
    data = get_data(100)
    result = trainer.predict(data, ignore_label=True)
    assert len(result) == 100*16

    
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
    result = trainer.save(filename)
    assert result == filename
    
    result = trainer.load(filename)
    assert result == filename
    os.remove(filename)

    filename1 = trainer.save()
    filename2 = trainer.load()
    os.remove(filename1)
    assert filename1 == filename2
    dir1 = os.path.dirname(filename1)
    os.rmdir(dir1)
    dir1 = os.path.dirname(dir1)
    os.rmdir(dir1)
    
    
class SmartMetric():
    def reset(self):
        pass
    
    def update(self, *args):
        pass
    
    def get(self):
        return "mymetric", 1.

    
def test_trainer_metrics():
    predictor = get_predictor()
    loss = F.mse_loss
    optim = torch.optim.Adam(predictor.parameters())
    model = Supervisor(predictor, loss)
    meter = MemoryMeter()
    trainer = Trainer(model, optim, meter, metrics=[SmartMetric()])
 
    data = get_data(100)
    trainer.run(data, data)
    assert trainer.epoch == 1

