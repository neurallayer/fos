# pylint: disable=E1101, C0116, C0114
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fos import Workout


def get_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1))


def get_data(steps):
    return [(torch.randn(16, 10), torch.rand(16, 1)) for i in range(steps)]


def test_trainer():
    model = get_model()
    loss = F.mse_loss
    optim = torch.optim.Adam(model.parameters())
    workout = Workout(model, loss, optim)

    data = get_data(100)
    workout.fit(data)
    assert workout.epoch == 1

    workout.fit(data, epochs=10)
    assert workout.epoch == 11

    valid_data = get_data(100)
    for minibatch in valid_data:
        workout.validate(*minibatch)
    assert workout.epoch == 11

    workout.fit(data, valid_data, epochs=5)
    assert workout.epoch == 16


def test_trainer_predict():

    model = get_model()
    loss = F.mse_loss
    optim = torch.optim.Adam(model.parameters())
    workout = Workout(model, loss, optim)

    data = torch.randn(16, 10)
    result = workout.predict(data)
    assert len(result) == 16


def test_trainer_state():
    model = get_model()
    loss = F.mse_loss
    workout = Workout(model, loss)

    state = workout.state_dict()
    workout = Workout(model, loss)
    workout.load_state_dict(state)

    filename = "./tmp_file.dat"
    result = workout.save(filename)
    assert result == filename

    result = workout.load(filename)
    assert result == filename
    os.remove(filename)

    filename1 = workout.save()
    filename2 = workout.load()
    os.remove(filename1)
    assert filename1 == filename2
    dir1 = os.path.dirname(filename1)
    os.rmdir(dir1)
    dir1 = os.path.dirname(dir1)
    os.rmdir(dir1)

def test_trainer_metrics():
    model = get_model()
    loss = F.mse_loss
    optim = torch.optim.Adam(model.parameters())
    workout = Workout(model, loss, optim)

    data = get_data(100)
    workout.fit(data, data)
    assert workout.epoch == 1
