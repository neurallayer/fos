# The the standard meters that are being provided. 
from fos.meters import * 


def test_printmeter():
    meter = PrintMeter()
    meter.update("loss", 0.5)
    meter.update("loss", 0.6)
    meter.update("loss", 0.7)
    
def test_notebookmeter():
    meter = NotebookMeter()
    meter.update("loss", 0.5)
    meter.update("loss", 0.6)
    meter.update("loss", 0.7)
    
def test_memorymeter():
    meter = MemoryMeter()
    meter.update("loss", 0.5)
    meter.display({"step":1})
    meter.update("loss", 0.6)
    meter.display({"step":2})
    meter.update("loss", 0.7)
    meter.display({"step":3})
    
    assert "loss" in meter.metrics
    assert "val_loss" not in meter.metrics
    
    steps, values = meter.get_history("loss")
    assert len(steps) == len(values)
    assert len(steps) == 3
    
    steps, values = meter.get_history("val_loss") 
    assert len(steps) == 0
    
    