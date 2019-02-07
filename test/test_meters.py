from unittest.mock import Mock

# The the standard meters that are being provided.
from fos.meters import PrintMeter, NotebookMeter, MemoryMeter, TensorBoardMeter


def init_meter(meter, steps):
    for i in range(steps):
        meter.update("loss", 1/(i+10))
        meter.display({"phase":"train", "step": i+1, "epoch":1})
    
    
def test_printmeter():
    meter = PrintMeter()
    cnt = 10
    init_meter(meter, cnt)   


def test_notebookmeter():
    meter = NotebookMeter()
    cnt = 10
    # init_meter(meter, cnt)  
    
    
def test_memorymeter():
    meter = MemoryMeter()
    cnt = 10
    init_meter(meter, cnt)

    assert "loss" in meter.metrics
    assert "val_loss" not in meter.metrics

    steps, values = meter.get_history("loss")
    assert len(steps) == len(values)
    assert len(steps) == cnt

    steps, values = meter.get_history("val_loss")
    assert len(steps) == 0
    
    meter.update("val_loss", 0.6)
    meter.display({"step": 3}) 
    steps, values = meter.get_history("val_loss")
    assert len(steps) == 1
    assert values[0] == 0.6
    
    state = meter.state_dict()
    assert len(state) > 0
    
    meter.reset()
    meter.load_state_dict(state)
    steps, values = meter.get_history("loss")
    assert len(steps) == len(values)
    assert len(steps) == cnt

    
class Nop():
    history = []
    
    def nop(self, *args, **kw):
        self.history.append((args, kw))
        
    def __getattr__(self, _): return self.nop
    
    
def test_tensorboardmeter():
    writer = Mock()
    meter = TensorBoardMeter(writer=writer)
    init_meter(meter, 10)
    writer.add_scalar.assert_called()
    meter.reset()
    assert meter.updated == {}
    init_meter(meter, 10)
    writer.add_scalar.assert_called()

    