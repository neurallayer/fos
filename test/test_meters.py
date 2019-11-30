from unittest.mock import Mock

# The standard meters that are being provided.
from fos.callbacks import PrintMeter, NotebookMeter, TensorBoardMeter
from fos import Workout
import torch.nn.functional as F
import torch.nn as nn


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


    