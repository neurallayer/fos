from unittest.mock import Mock

# The standard meters that are being provided.
from fos.meters import PrintMeter, MultiMeter, NotebookMeter, MemoryMeter, TensorBoardMeter


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
    
def test_multimeter():
    meter = MultiMeter(MemoryMeter(), MemoryMeter())
    cnt = 10
    init_meter(meter, cnt) 
    state = meter.state_dict()
    assert len(state) == 2

def test_notebookmeter():
    meter = NotebookMeter()
    meter.tqdm = Mock()
    cnt = 10
    init_meter(meter, cnt)
    state = meter.state_dict()
    assert state == None

    
def test_memorymeter():
    meter = MemoryMeter()
    cnt = 10
    init_meter(meter, cnt)


    steps, values = meter.get_history("mymetric")
    assert len(steps) == len(values)
    assert len(steps) == cnt

    steps, values = meter.get_history("mymetric_")
    assert len(steps) == 0
    
    state = meter.state_dict()
    assert len(state) > 0
    
    meter.reset()
    meter.load_state_dict(state)
    steps, values = meter.get_history("mymetric")
    assert len(steps) == len(values)
    assert len(steps) == cnt

    
def _test_tensorboardmeter():
    writer = Mock()
    meter = TensorBoardMeter(writer=writer)
    init_meter(meter, 10)
    writer.add_scalar.assert_called()
    meter.reset()
    init_meter(meter, 10)
    writer.add_scalar.assert_called()

    