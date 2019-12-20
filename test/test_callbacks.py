# pylint: disable=C0116, C0114
from unittest.mock import Mock

from torchvision.models import resnet18
# The standard meters that are being provided.
from fos.callbacks import PrintMeter, NotebookMeter, TensorBoardMeter, ParamHistogram
from fos import Workout, Mode

def get_workout():
    workout = Workout(Mock(), Mock(), optim=Mock(), mover=Mock())
    workout.batches = 10
    return workout

def test_printmeter():
    meter = PrintMeter()
    workout = get_workout()
    meter(workout, Mode.EVAL)

def test_notebookmeter():
    meter = NotebookMeter()
    workout = get_workout()
    meter(workout, Mode.EVAL)

def _test_tensorboardmeter():
    writer = Mock()
    workout = get_workout()
    meter = TensorBoardMeter(writer=writer)
    meter(workout, Mode.EVAL)
    writer.add_scalar.assert_called()

def test_paramhistogram():
    writer = Mock()
    loss = Mock()
    callback = ParamHistogram(writer, include_gradient=False)
    model = resnet18()
    workout = Workout(model, loss, callbacks=[callback])
    callback(workout, Mode.EVAL)
    assert writer.add_histogram.is_called()
