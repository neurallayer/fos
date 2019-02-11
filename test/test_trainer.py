import torch
import torch.nn as nn
import torch.nn.functional as F
from fos import Supervisor, Trainer
from fos.meters import MemoryMeter


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
    
    batch_size = 16
    test_data = [torch.randn(batch_size, 10) for i in range(20)]
    result = trainer.predict(test_data)
    assert len(result) == len(test_data)*batch_size