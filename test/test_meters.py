from unittest.mock import Mock

# The standard meters that are being provided.
from fos.callbacks import PrintMeter, NotebookMeter, TensorBoardMeter


class SmartMetric():
    def reset(self):
        pass
    
    def update(self, *args):
        pass
    
    def get(self):
        return "mymetric", 1.


def init_meter(meter, steps):
    metric = SmartMetric()
    for i in range(steps):
        meter.display([metric], {"phase":"train", "step": i+1, "epoch":1, "progress":0.5})
    
    
def test_printmeter():
    meter = PrintMeter()
    cnt = 10
    init_meter(meter, cnt) 
    state = meter.state_dict()
    assert state == None
    

def test_notebookmeter():
    meter = NotebookMeter()
    meter.tqdm = Mock()
    cnt = 10
    init_meter(meter, cnt)
    state = meter.state_dict()
    assert state == None

    

def _test_tensorboardmeter():
    writer = Mock()
    meter = TensorBoardMeter(writer=writer)
    init_meter(meter, 10)
    writer.add_scalar.assert_called()
    meter.reset()
    init_meter(meter, 10)
    writer.add_scalar.assert_called()

    