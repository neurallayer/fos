from .accuracy import *
from .confusion import *
from .modelmetrics import *


class Meter():
    
    def reset(self):
        pass
    
    def add(self, value, n=1):
        pass
    
    def calc(self):
        pass
    

class AvgMeter():
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._sum = 0
        self._n = 0
    
    def add(self, value, n=1):
        self._sum += value
        self._n += n
    
    def calc(self):
        if self._n == 0:
            return None
        else:
            return (self._sum/self._n).item()
    
class EvalMetric():
    
    def reset(self):
        pass
    
    def update(self, y_pred, y_true):
        pass
    
    def get(self):
        pass
    


class MEA(EvalMetric):
    
    def __init__(self, meter=None):
        self.meter = AvgMeter() if meter is None else meter 
        self.name = "mae"
    
    def reset(self):
        self.meter.reset()
    
    def update(self, y_pred, y_true):
        value = (y_pred - y_true).abs().mean()
        self.meter.add(value)
    
    def get(self):
        return self.name, self.meter.calc()


class LossMetric(EvalMetric):
    
    def __init__(self, meter=None):
        self.meter = AvgMeter() if meter is None else meter 
        self.name = "loss"
    
    def reset(self):
        self.meter.reset()
    
    def update(self, loss, _):
        self.meter.add(loss)
    
    def get(self):
        return self.name, self.meter.calc()


    
class BinaryAccuracy(EvalMetric):
    '''Calculate the binary accuracy score between the predicted and target values.

    Args:
        threshold (float): The threshold to use to determine if the input value is 0 or 1
        sigmoid (bool): should sigmoid activatin be applied to the input
    '''

    def __init__(self, threshold=0., sigmoid=False):
        self.sigmoid = sigmoid
        self.threshold = threshold
        self.meter = AvgMeter()
        self.name = "accuracy"

        
    def reset(self):
        self.meter.reset()

        
    def update(self, input, target):
        input = input.flatten(1)
        target = target.flatten(1)

        assert input.shape == target.shape, "Shapes of target and predicted values should be same"

        if self.sigmoid:
            input = input.sigmoid()

        input = (input > self.threshold).int()
        target = target.int()

        self.meter.add((input == target).float().mean())

    def get(self):
        return self.name, self.meter.calc()
