import time
import logging
from abc import abstractmethod
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import logging

from ..calc import AvgCalc

class Meter():
    '''This is the interface that needs to be implemented
       by any meter.

       The iteraction is as follows:

            1. Whenever the trainer receives metrics from the model, it updates
            the meter with the received metrics. So this is both during training
            as well as validation after each iteration.

            2. Whenever a step (= an iteration that updates the model) has finished,
            the trainer will also call the display method to give the meter the opportunity to
            display (or process in any other way) the metrics it has captured so far.
            For the training phase this is once after every step, for the validaiton
            only once per epoch since the step counter doesn't change during validation.

       See also the BaseMeter that provides a sensible default implementation for many of
       the methods defined in this interface.
    '''

    @abstractmethod
    def reset(self):
        '''Reset the state kept by the meter. If a meter uses calculators
           the meter will typically also invoke calculator.clear() to ensure they
           are also cleared.
        '''

    @abstractmethod
    def update(self, key, value):
        '''update the state of the meter with a certain metric and its value.

           Args:
               key (str): the name of the metric
               valaue: the value of the metric
        '''

    @abstractmethod
    def display(self, ctx):
        '''display the values of the meter. Display can visual (to standard out or a notebook),
           but also to a file or database.

           Args:
               ctx (dict): The context including values for steps and epochs.
        '''

    def state_dict(self):
        '''get the internal state of the meter. The default implementation
        assumes the meter is stateless.
        '''
        return None

    def load_state_dict(self, state):
        '''Load a previous state into the meter. The default implementation
        assumes the meter is stateless and just invokes reset.

        Args:
            state: the state to be loaded
        '''
        self.reset()


class MultiMeter(Meter):
    '''Container of other meters, allowing for more than one meter
       be used during training.

       Arguments:
           meters: the meters that should be wrapped

       Example usage:
       
       .. code-block:: python
       
           meter1 = NotebookMeter()
           meter2 = TensorBoardMeter()
           meter = MultiMeter(meter1, meter2)
           trainer = Trainer(model, optim, meter)
    '''

    def __init__(self, *meters):
        self.meters = list(meters)

    def add(self, meter):
        '''Add a meter to this multimeter'''
        self.meters.append(meter)

    def update(self, key, value):
        for meter in self.meters:
            meter.update(key, value)

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def display(self, ctx):
        for meter in self.meters:
            meter.display(ctx)
            
    def state_dict(self):
        return [meter.state_dict() for meter in self.meters]

    def load_state_dict(self, state):
        if len(state) != len(self.meters):
            logging.warning("Invalid state receive for MultiMeter, will reset only.")
            self.reset()
        else:
            for meter_state, meter in zip(state, self.meters):
                meter.load_state_dict(meter_state)
            


class BaseMeter(Meter):
    '''Base meter that provides a default implementation for the various methods except 
       the display method. So a subclas has to implement the display method.

       Behaviour rules when creating an instance:
           1. If nothing is specified, it will record all metrics using the average
           calculator.

            2. If only one or more exclude metrics are specifified, it will record all 
            metrics except the ones listed in the exclude argument.

            3. If metrics are provided, only record those metrics and ignore other metrics.
            The exclude argument is also ignored in this case.

       Args:
           metrics (dict): the metrics and their calculators that should be
               handled. If this argument is provided, metrics not mentioned
               will be ignored by this meter. If no value is provided, the
               meter will handle all the metrics.
           exclude (list): list of metrics to ignore

       Example usage:
       
       .. code-block:: python
       
           meter = some_meter()
           meter = some_meter(metrics={"acc": MomentumMeter(), "val_acc": AvgMeter()})
           meter = some_meter(exclude=["vall_loss"])
    '''

    def __init__(self, metrics=None, exclude=None):
        self.metrics = metrics if metrics is not None else OrderedDict()
        self.exclude = exclude if exclude is not None else []
        self.dynamic = True if metrics is None else False
        self.updated = {}

    def _get_calc(self, key):
        if key in self.metrics:
            return self.metrics[key]

        if key in self.exclude:
            return None

        if self.dynamic:
            self.metrics[key] = AvgCalc()
            return self.metrics[key]

        return None

    def update(self, key, value):
        calc = self._get_calc(key)
        if calc is not None:
            calc.add(value)
            self.updated[key] = True

    def reset(self):
        for calculator in self.metrics.values():
            calculator.clear()
        self.updated = {}


class NotebookMeter(Meter):
    '''Meter that displays the metrics and progress in
       a Jupyter notebook. This meter uses tqdm to display
       the progress bar.
    '''

    def __init__(self):
        self.tqdm = None
        self.reset()
        self.bar_format = "{l_bar}{bar}|{elapsed}<{remaining}"

    def _get_tqdm(self):
        if self.tqdm is None:
            self.tqdm = tqdm(
                total=100,
                mininterval=1,
                bar_format=self.bar_format)
        return self.tqdm

    def update(self, key, value):
        if key not in self.calculators:
            self.calculators[key] = AvgCalc()

        calc = self.calculators[key]
        calc.add(value)

    def format(self, key, value):
        try:
            value = float(value)
            result = "{}={:.5f} ".format(key, value)
        except BaseException:
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

        pb = self._get_tqdm()
        progress = ctx["progress"]
        if progress is not None:
            rel_progress = int(progress * 100) - self.last
            if rel_progress > 0:
                pb.update(rel_progress)
                self.last = int(progress * 100)
            pb.set_description(result, refresh=False)
        else:
            pb.set_description(result)


class PrintMeter(BaseMeter):
    '''Displays the metrics by using a simple print
       statement.

       If you use this is a shell script, please be aware that
       by default Python buffers the output. You can change this
       behaviour by using the `-u` option. See also:

       `<https://docs.python.org/3/using/cmdline.html#cmdoption-u>`_

       Args:
           throttle (int): how often to print output, default is oncr every 3 seconds.

    '''

    def __init__(self, metrics=None, exclude=None, throttle=3):
        super().__init__(metrics, exclude)
        self.throttle = throttle
        self.next = -1

    def reset(self):
        super().reset()
        self.next = -1

    def _format(self, key, value):
        try:
            value = float(value)
            result = "{}={:.6f} ".format(key, value)
        except BaseException:
            result = "{}={} ".format(key, value)
        return result

    def display(self, ctx):
        now = time.time()
        if now > self.next:
            result = "{}:[{:6}] => ".format(ctx["phase"], ctx["step"])
            for key, calculator in self.metrics.items():
                if key in self.updated:
                    value = calculator.result()
                    if value is not None:
                        result += self._format(key, value)
            print(result)
            self.updated = {}
            self.next = now + self.throttle


            

class MemoryMeter(BaseMeter):
    '''Meter that stores values in memory for later use.
       With the get_history method the values for a metric
       can be retrieved.

       Since it stores everything in memory, this meter should be used 
       with care in order to avoid memory issues.
    '''

    def __init__(self, metrics=None, exclude=None):
        super().__init__(metrics, exclude)
        self.history = []

    def get_history(self, name, min_step=0):
        '''Get the history for one of the stored metrics.

           Arguments:
               name: the name of the metric
               min_step: the minimum step where to start collecting the
                stored history.
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

    def state_dict(self):
        return self.history

    def load_state_dict(self, state):
        self.history = state


class TensorBoardMeter(BaseMeter):
    '''Log the metrics to a tensorboard file so they can be reviewed
       in tensorboard. Currently supports the following type for metrics:

       * string, not a common use case. But you could use it to log some remarks::

               meter = TensorBoardMeter(metrics={"acc":AvgCalc(), "remark": RecentCalc()})
               ...
               meter.update("remark", "Some exception occured, not sure about impact")

       * dictionary of floats or strings. Every key in the dictionary will be 1 metric
       * dist of float or strings. Every element in the list will be 1 metric
       * float or values that convert to a float. This is the default if the other ones don't apply.
         In case this fails, the meter ignores the exception and the metric will not be logged.
         
         
      Args:
          writer: the writer to use for logging
          prefix: any prefix to add to the metric name. This allows for metrics to be 
            grouped together in Tensorboard.
            
       Example usage:
       
       .. code-block:: python
      
          writer = HistoryWriter("/tmp/runs/myrun")
          metrics = {"loss":AvgCalc(), "val_loss":AvgCalc()}
          meter = TensorBoardMeter(writer, metrics=metrics, prefix="metrics/")
          ...
    '''

    def __init__(self, writer=None, metrics=None, exclude=None, prefix=""):
        super().__init__(metrics, exclude)
        self.writer = writer
        self.prefix = prefix

    def set_writer(self, writer):
        '''Set the writer to use for logging the metrics'''
        self.writer = writer

    def _write(self, name, value, step):
        if isinstance(value, dict):
            for k, v in value.items():
                self._write(name + "/" + k, v, step)
            return

        if isinstance(value, str):
            self.writer.add_text(name, value, step)
            return

        if isinstance(value, list):
            for idx, v in enumerate(value):
                self._write(name + "/" + str(idx + 1), v, step)
            return

        try:
            value = float(value)
            self.writer.add_scalar(name, value, step)
        except BaseException:
            logging.warning("ignoring metric %s", name)

    def display(self, ctx):
        for key, calculator in self.metrics.items():
            if key in self.updated:
                value = calculator.result()
                if value is not None:
                    full_name = self.prefix + key
                    self._write(full_name, value, ctx["step"])
        self.updated = {}




class VisdomMeter(BaseMeter):
    '''Visdom meter.
    
       TODO: currently not functional.
    '''

    def __init__(self, vis=None, metrics=None, exclude=None, prefix=""):
        super().__init__(metrics, exclude)
        self.vis = vis
        self.prefix = prefix
        self.graphs = {}
        
    def _write(self, name, value, step):
        update='append'
        if name not in self.graphs:
            self.graphs = vis.line
            
        graph = self.graphs[name] 
        graph(x=step, y=value, update=update)
        

    def display(self, name, step):
        for key, calculator in self.calculators.items():
            if key in self.updated:
                value = calculator.result()
                if value is not None:
                    full_name = self.prefix + key
                    self._write(full_name, value, step)
        self.updated = {}
