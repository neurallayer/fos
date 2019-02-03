import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict 

from ..calc import AvgCalc

class Meter():
    '''This is the meter interface that needs to be implemented 
       by any meter. For an example how to implement a custom meter, 
       look at the BaseMeter class.
    '''
   
    def reset(self):
        '''reset the state kept by the meter'''
        ...

    def update(self, key, value):
        '''update the state of the meter'''
        ...
        
    def display(self, ctx):
        '''display the values of the meter'''
        ...
        
        
class MultiMeter(Meter):
    '''Container of other meters, allowing for more than one meter
       be used during training.
       
       Arguments:
           meters: the meters that should be wrapped
       
       Example:
           meter = MultiMeter(NotebookMeter(), TBMeter())
           trainer = Trainer(model, optim, meter)
    '''

    def __init__(self, *meters):
        self.meters = meters
        
    def add(self, meter):
        '''Add a meter to this multimeter'''
        self.meters.append(meter)
        
    def update(self, key:str, value):
        for meter in self.meters:
            meter.update(key, value)
    
    def reset(self):
        for meter in self.meters:
            meter.reset()
         
    def display(self, ctx):
        for meter in self.meters:
            meter.display(ctx)

        
        
class BaseMeter(Meter):
    '''Base meter that provides some sensible default implementations
       for the various methods except the display method.
       
       So many of the subclasses only have to implement the display method
    '''
    
    def __init__(self, calculators={}):
        self.calculators = calculators
        self.updated = False
        
    def update(self, key:str, value):
        if key in self.calculators:
            calc = self.calculators[key]
            calc.add(value)
            self.updated = True
    
    def reset(self):
        for calculator in self.calculators.values(): 
            calculator.clear()
        self.updated = False
        
    def display(self, ctx):
        ...


class NotebookMeter(Meter):
    '''Meter that display the progress in a Jupyter notebook.
    '''
    
    def __init__(self):
        self.tqdm = None
        self.reset()
        self.bar_format = "{l_bar}{bar}|{elapsed}<{remaining}"
        
    def get_tqdm(self):
        if self.tqdm is None:
            self.tqdm = tqdm(total=100, mininterval=1, bar_format=self.bar_format)
        return self.tqdm
        
    def update(self, key:str, value):
        if key not in self.calculators:
            self.calculators[key] = AvgCalc()
        
        calc = self.calculators[key]  
        calc.add(value)
        
    def format(self, key:str, value):
        try:
            value = float(value)
            result = "{}={:.5f} ".format(key, value)
        except:
            result = "{}={} ".format(key, value)        
        return result
    
    def reset(self):
        self.calculators = OrderedDict()
        self.last = 0
        if self.tqdm is not None:
            self.tqdm.close()
            self.tqdm = None
   
    def display(self, ctx):    
        result = "[{:3}:{:6}] ".format(ctx["epoch"], ctx["step"])
        for key, calculator in self.calculators.items():
            value = calculator.result()
            if value is not None:
                result += self.format(key, value)
        
        tqdm = self.get_tqdm()
        progress = ctx["progress"]
        if progress is not None:
            rel_progress = int(progress*100) - self.last
            if rel_progress > 0: 
                tqdm.update(rel_progress)
                self.last = int(progress*100)
            tqdm.set_description(result, refresh=False)
        else:
            tqdm.set_description(result)
                
class PrintMeter(BaseMeter):
    
    def __init__(self, calculators={}, throttle=3):
        super().__init__(calculators)
        self.throttle = throttle
        self.next = -1
    
    def reset(self):
        super().reset()
        self.next = -1
    
    def format(self, key:str, value):
        try:
            value = float(value)
            result = "{}={:.6f} ".format(key, value)
        except:
            result = "{}={} ".format(key, value)        
        return result
    
    def display(self, ctx):
        if not self.updated: return
        now = time.time()
        if now > self.next: 
            result = "{}:[{:6}] => ".format(ctx["phase"], ctx["step"])
            for key, calculator in self.calculators.items():
                value = calculator.result()
                if value is not None:
                    result += self.format(key, value)
            print(result)
            self.next = now + self.throttle

        
class TensorBoardMeter(BaseMeter):
    
    def __init__(self, calculators={}, writer=None, prefix=""):
        super().__init__(calculators)
        self.writer = writer
        self.prefix = prefix
         
    def set_writer(self, subdir):
        self.writer = SummaryWriter("/tmp/runs/" + subdir)    
            
    def write(self, name, value, step):
       
        if isinstance(value, dict):
            for k,v in value.items():
                self.write(name+":"+k, v, step)
        else:
            try:
                value = float(value)
                self.writer.add_scalar(name, value, step)
            except:
                pass

    def display(self, ctx):
        if not self.updated: return
        for key, calculator in self.calculators.items():
            value = calculator.result()
            if value is not None:
                full_name = self.prefix + key
                self.write(full_name, value, ctx["step"])


        
class VisdomMeter(BaseMeter):
    
    def __init__(self, calculators={}, vis=None, prefix=""):
        super().__init__(calculators)
        self.vis = visdom.Visdom() if vis is None else vis
        self.prefix = prefix
            
            
    def write(self, name, value, step):
        pass # TODO

    def display(self, name, step):
        if not self.updated: return
        for key, calculator in self.calculators.items():
            value = calculator.result()
            if value is not None:
                full_name = self.prefix + key
                self.write(full_name, value, step)


class BaseMeter2(Meter):
    '''Base meter that provides some sensible default implementations
       for the various methods except the display method.
       
       So many of the subclasses only have to implement the display method.
       
       Arguments:
           metrics (dict): the metrics and their calculators that should be 
               handled. If this argument is provided, metrics not mentioned
               will be ignored by this meter. If no value is provided, the 
               meter will handle all the metrics.
               
    '''
    
    def __init__(self, metrics=None):
        self.metrics = metrics if metrics is not None else OrderedDict()
        self.dynamic = True if metrics is None else False
        self.updated = {}
        
    def get_calc(self, key):
        if key in self.metrics:
            return self.metrics[key]
        
        if self.dynamic:
            self.metrics[key] = AvgCalc()
            return self.metrics[key]
        
        return None
            
    def update(self, key:str, value):
        calc = self.get_calc(key)
        if calc is not None:
            calc.add(value)
            self.updated[key] = True
    
    def reset(self):
        for calculator in self.metrics.values(): 
            calculator.clear()
        self.updated = {}
        
    def display(self, ctx):
        ...


                
class MemoryMeter(BaseMeter2):
    '''Meter that stores values in memory for later use.
       With the get_history method the values for a metric
       can be retrieved. 
       
       Since it stores everything in memory, should be used with care
       in order to avoid out of memory issues.
    '''
    
    def __init__(self, metrics=None):
        super().__init__(metrics)
        self.history = []
       
    def get_history(self, name, min_step=0):
        '''Get the history for one of the metrics
        
           Arguments:
               name: the name of the metric
        '''
        result = []
        steps = []
        for (step, key, value) in self.history:
            if (step >= min_step) and (name == key) and (value is not None):
                result.append(value)
                steps.append(step)
        return steps, result
    
    def display(self, ctx):
        for key, calculator in self.metrics.items():
            if key in self.updated:
                value = calculator.result()
                if value is not None:
                    self.history.append((ctx["step"], key, value))
        self.updated = {}
        
        
            


