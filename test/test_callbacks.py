# pylint: disable=C0116, C0114
from unittest.mock import Mock
import torch.nn.functional as F
import torch.nn as nn

# The standard meters that are being provided.
from fos.callbacks import PrintMeter, NotebookMeter, TensorBoardMeter, ParamHistogram
from fos import Workout


def get_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1))

def get_workout():
    workout = Workout(get_model(), F.mse_loss)
    workout.batches = 10
    return workout

def test_printmeter():
    meter = PrintMeter()
    workout = get_workout()
    meter(workout, "valid")

def test_notebookmeter():
    meter = NotebookMeter()
    workout = get_workout()
    meter(workout, "valid")

def _test_tensorboardmeter():
    writer = Mock()
    workout = get_workout()
    meter = TensorBoardMeter(writer=writer)
    meter(workout, "valid")
    writer.add_scalar.assert_called()

def test_paramhistogram():
    writer = Mock()
    loss = Mock()
    callback = ParamHistogram(writer, include_gradient=False)
    workout = Workout(get_model(), loss, callbacks=callback)
    callback(workout, "valid")
    assert writer.add_histogram.is_called()
