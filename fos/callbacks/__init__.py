'''Contains various callbacks that can be used during training to add a wide variety
of functionality
'''

import logging
from abc import abstractmethod
from collections import OrderedDict
from tqdm import tqdm
# pylint: disable=R0903


def _get_metrics2process(workout, metrics: [str]):
    return [m for m in metrics if workout.has_metric(m)]


class Callback():
    '''This is the interface that needs to be implemented
       for classes that want to function as a callback.

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
    def __call__(self, workout, phase: str):
        '''update the state of the meter with a certain metric and its value.

           Args:
               workout: the workout
               phase: the phase ("trian" or "valid")
        '''


class EarlyStop(Callback):
    '''Automatically stop the training if a certain metric doesn't improve anymore.
       This is checked at the end of every epoch.
    '''

    def __init__(self, metric="val_loss", minimize=True):
        self.metric = metric
        self.minimize = minimize
        self.value = float('Inf') if minimize else -float('Inf')

    def __call__(self, workout, phase: str):
        if phase == "train":
            return

        value = workout.get_metric(self.metric)

        if self.minimize & value < self.value:
            self.value = value
        elif not self.minimize & value > self.value:
            self.value = value
        else:
            workout.stop()


class AutoSave(Callback):
    '''Automatically save the model as long as a certain metric improves. This
       is run at the end of every epoch.
    '''

    def __init__(self, metric="val_loss", minimize=True, filename=None):
        self.metric = metric
        self.minimize = minimize
        self.value = float('Inf') if minimize else -float('Inf')
        self.filename = filename

    def __call__(self, workout, phase: str):
        if phase == "train":
            return

        value = workout.get_metric(self.metric)

        if self.minimize & value < self.value:
            self.value = value
            workout.save(self.filename)
        elif not self.minimize & value > self.value:
            self.value = value
            workout.save(self.filename)
        else:
            workout.stop()


class RegisterLR(Callback):
    '''Get the learning rates used by the optimizer. Comes in handy
       if you use for example a scheduler and want to track how it
       changed the learning rate during the training. This will add the
       learning rates to the history of the workout.

       This callback supports optimizers with multiple parameter groups
    '''

    def __init__(self, prefix="lr_"):
        self.prefix = prefix

    def __call__(self, workout, phase: str):
        for idx, group in enumerate(workout.optim.param_groups):
            name = self.prefix + idx
            value = group["lr"]
            workout.update_history(name, value)


class EpochSave(Callback):
    '''Save the model at the end of every epoch.
    '''

    def __init__(self, filename=None):
        self.filename = filename

    def __call__(self, workout, phase: str):
        if phase == "train":
            return
        workout.save(self.filename)


class SilentMeter(Callback):
    '''Silently ignore all the metrics and don't produce any output'''

    def __call__(self, workout, phase):
        pass



class BaseMeter(Callback):
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
        self.metrics = metrics
        self.exclude = exclude

    def get_metrics(self, workout):
        '''Get the metrics to process'''

        if self.metrics is None:
            metrics = workout.get_metrics()
            if self.exclude is not None:
                metrics = list(filter(lambda m: m not in self.exclude, metrics))
        else:
            metrics = self.metrics
        return metrics

    @abstractmethod
    def __call__(self, workout, phase: str):
        pass


class PrintMeter(BaseMeter):
    '''Displays the metrics by using a simple print
       statement the end of an epoch

       If you use this in a shell script, please be aware that
       by default Python buffers the output. You can change this
       behaviour by using the `-u` option. See also:

       `<https://docs.python.org/3/using/cmdline.html#cmdoption-u>`_

       Args:
           metrics: which metrics should be printed.

    '''
    def _format(self, key, value):
        try:
            value = float(value)
            result = " - {} : {:.6f} ".format(key, value)
        except ValueError:
            result = " - {} : {} ".format(key, value)
        return result

    def __call__(self, workout, phase: str):
        if phase != "valid":
            return
        result = "{:6}:{:6}".format(workout.epoch, workout.step)
        for metric in self.get_metrics(workout):
            value = workout.get_metric(metric)
            if value is not None:
                result += self._format(metric, value)
        print(result)





class NotebookMeter(Callback):
    '''Meter that displays the metrics and progress in
       a Jupyter notebook. This meter uses tqdm to display
       the progress bar.
    '''

    def __init__(self, metrics=None):
        self.tqdm = None
        self.epoch = -1
        self.metrics = metrics if metrics is not None else ["loss", "acc", "val_loss", "val_acc"]
        self.bar_format = "{l_bar}{bar}|{elapsed}<{remaining}"

    def _get_tqdm(self, workout):
        if self.tqdm is None:
            self.tqdm = tqdm(
                total=workout.batches+1,
                mininterval=1,
                bar_format=self.bar_format)
        return self.tqdm

    def _format(self, key, value):
        try:
            value = float(value)
            result = "{}={:.5f} ".format(key, value)
        except ValueError:
            result = "{}={} ".format(key, value)
        return result

    def _new_meter(self, workout):
        # self.last = workout.step - 1
        self.epoch = workout.epoch
        if self.tqdm is not None:
            self.tqdm.close()
            self.tqdm = None

    def __call__(self, workout, phase):
        if workout.epoch > self.epoch:
            self._new_meter(workout)

        # progress = (workout.step - self.last)/workout.batches

        result = "[{:3}:{:6}] ".format(workout.epoch, workout.step)
        for metric in self.metrics:
            if workout.has_metric(metric):
                result += self._format(metric, workout.get_metric(metric))

        progressbar = self._get_tqdm(workout)
        progressbar.update(1)
        # pb.set_description(result, refresh=False)
        progressbar.set_description(result)


class TensorBoardMeter(Callback):
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
          metrics = ["loss", "acc", "val_acc"]
          meter = TensorBoardMeter(writer, metrics=metrics, prefix="metrics/")
          ...
    '''

    def __init__(self, writer=None, metrics=None, prefix=""):
        super().__init__()
        self.writer = writer
        self.metrics = metrics if metrics is not None else ["loss", "acc", "val_loss", "val_acc"]
        self.prefix = prefix

    def set_writer(self, writer):
        '''Set the writer to use for logging the metrics'''
        self.writer = writer

    def __call__(self, workout, phase):
        for metric in self.metrics:
            if workout.has_metric(metric):
                value = workout.get_metric(metric)
                if value is not None:
                    name = self.prefix + metric
                    try:
                        value = float(value)
                        self.writer.add_scalar(name, value, workout.step)
                    except ValueError:
                        logging.warning("ignoring metric %s", name)



class ParamHistogram(Callback):
    '''Make histograms of the weights and gradients of the parameters
       in the model. This metric writes directly to a tensorboard file and
       doesn't support Meters to perform that task.

       Every layer will get its own histograms. For example a Linear
       layer will get 4 histograms by default:

       - 2 Histograms for the bias parameter (values + gradient)
       - 2 Histograms for the weight parameter (values + gradient)

       The histograms provide insights into the model and helps to expose issues
       like vanishing gradients. This metric uses directly TensorBoardX to
       create the files that contain histograms. So you'll need TensorBoard to
       view them.

       This metric can generate a lot of data, especially if you have many layers in your
       model. So typically you don't run this every step (see also the skip argument below).

       Arguments:
           writer: which Tensorboard writer to use.
           prefix (str): do you want to groep the metrics under a common "card" within tensorboard.
           skip (int): how many steps to skip until the next histograms are generated. If
           you run this metric every step it will slown down the training. Default is 500
           include_weight (bool): Should it include the weights in the histograms
           include_gradient (bool): Should it include the gradients int the histograms
    '''

    def __init__(self, writer=None, prefix="", skip=500,
                 include_weight=True, include_gradient=True):
        self.writer = writer
        self.prefix = prefix
        self.skip = skip
        self.include_gradient = include_gradient
        self.include_weight = include_weight

    def set_writer(self, writer):
        '''Set the summary writer to use to output the metrics'''
        self.writer = writer

    @staticmethod
    def _get_np(param):
        return param.clone().detach().cpu().numpy()

    def __call__(self, workout, phase):

        model = workout.model

        if (model.step % self.skip) != 0:
            return

        for param_name, param in model.named_parameters():

            if self.include_weight:
                name = "weight/" + param_name
                self._write(name, param.data, workout.step)

            if self.include_gradient and hasattr(param, "grad"):
                name = "gradient/" + param_name
                self._write(name, param.grad, workout.step)

    def _write(self, name, value, step):
        value = self._get_np(value)
        name = self.prefix + name
        self.writer.add_histogram(name, value, step)
