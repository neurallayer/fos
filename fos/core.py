'''
The core module contains the basic functionality required to train a model.

There are 3 classes in the core package:

1) Workout, that enables training a model with just a few lines of code
2) Mover, that will move tensors to the desired device like a GPU
3) SmartHistory, that will store the captured metrics

'''

import time
import os
import logging
from enum import Enum
from typing import Callable, Iterable, Tuple, List, NoReturn, Type
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

# pylint: disable=W0622


class Mode(Enum):
    '''Identifies the phase the training is in. The two many phases are TRAIN and EVAL.
       The value of each phase is used to prepand to the metric name to make it unique.
    '''

    TRAIN = ""
    EVAL = "val_"
    OTHER = "other_"

# pylint: disable=W0223
class Workout(nn.Module):
    '''Coordinates all the training of a model and provides many methods that reduces the amount
       of boilerplate code required when training a model. In its the simplest form it can be
       used as follows:

       .. code-block:: python

           workout = Workout(mymodel, F.mse_loss)
           workout.fit(data, epochs=10)

       Besides the added loss function and optional optimizer, also additional metrics can be
       specified to get insights into the performance of model.

       Args:
           model (nn.Module): The model that needs to be trained.
           loss_fn (function): The loss function (objective) that should be used to train the model
           optim: The optimizer to use. If none is provided, SGD will be used
           mover: the mover to use. If None is specified, a default mover will be created to move
           tensors to the correct device.
           metrics : The metrics that should be evaluated during the training and validation.
           Every metric can be specified as an optional argument, e.g acc=BinaryAccuracy()

       Example usage:

       .. code-block:: python

           workout = Workout(model, F.mse_loss, Adam(model.paramters()), acc=BinaryAccuracy())
    '''
    # pylint: disable= R0902
    def __init__(self, model: nn.Module, loss_fn: Callable,
                 optim: Optimizer = None, mover=None, **metrics):
        super().__init__()
        self.model = model
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.mover = mover if mover is not None else Mover.get_default(model)
        self.history = {}
        self.step = 0
        self.epoch = 0
        self.batches = None
        self._id = str(int(time.time()))
        self.optim = optim if optim is not None else torch.optim.SGD(
            model.parameters(), lr=1e-3)

    class StopError(Exception):
        '''used internally to stop the training before all the epochs have finished'''

    def update_history(self, name: str, value: float) -> None:
        ''''Update the history for the passed metric name and its value. It will store
            the metric under the current step.
        '''
        if name not in self.history:
            self.history[name] = SmartHistory()
        self.history[name][self.step] = value

    def get_metricname(self, name: str, phase: Mode) -> str:
        '''Get the fully qualified name for a metric. If phase equals train the
           metric name is as specified and if phase is "valid" the metric name
           is "val_" + name.

           So for example wehn performing a trianing cycle with validaiton step
           the following two loss metrics will be availble:

           "loss" : for the recorded loss during training
           "val_loss": or the recorded loss during validation
        '''
        # pylint: disable= R0201
        return phase.value + name

    def _invoke_metrics(self, loss, pred, target, phase: Mode) -> None:
        '''Run the configured metrics functions and update the history with theirs results.

           Args:
               loss (scaler): the loss value
               pred (Tensor): the predicted value
               target (Tensor): the target value
        '''
        loss_name = self.get_metricname("loss", phase)
        self.update_history(loss_name, loss.detach())
        for name, func in self.metrics.items():
            value = func(pred, target)
            fqname = self.get_metricname(name, phase)
            if hasattr(value, "detach"):
                value = value.detach()
            self.update_history(fqname, value)

    def get_metrics(self) -> List[str]:
        '''Get all metrics that have at least one value logged'''
        return self.history.keys()

    def has_metric(self, name: str, step: int = None) -> bool:
        '''Check for the metric value for the provided metric name and step.
           True of it exist, False otherwise.
        '''
        if name not in self.history:
            return False
        step = step if step is not None else self.step
        return step in self.history[name]

    def get_metric(self, name: str, step: int = None):
        '''Get the metric value for the provided metric name and step.
            If no step is provided, the last workout step is used.
        '''
        step = step if step is not None else self.step
        return self.history[name][step]

    def forward(self, *minibatch) -> Tuple:
        '''Implementation of the forward method in nn.Module

           Args:
               input (Tensor): the input data for the model
               target (Tensor): the target data for the loss function
        '''
        input, target = minibatch
        pred = self.model(input)
        loss = self.loss_fn(pred, target)
        return loss, pred

    def trace(self, input, check_trace=False):
        '''Create a traced model and return it.

           Args:
               input (Tensor): a minibatch of input tensors
        '''
        self.model.train()
        with torch.set_grad_enabled(False):
            input = self.mover(input)
            traced_model = torch.jit.trace(self.model, input, check_trace=check_trace)
            return traced_model

    def predict(self, input):
        '''Predict a batch of data at once and return the result. No metrics
           will be generated when predicting values. The data will be moved to
           the device using the configured mover.

           Args:
               input (Tensor): the minibatch of only the input tensors
        '''
        self.model.eval()
        with torch.set_grad_enabled(False):
            input = self.mover(input)
            pred = self.model(input)
            return pred

    def validate(self, *minibatch) -> None:
        '''Perform a single validation iteration. If there are metrics
           configured, they will be invoked and the result is returned together
           with the loss value. The data will be moved to the
           device using the configured mover.

           Args:
               minibatch: the input and target tensor
        '''
        self.model.eval()
        with torch.set_grad_enabled(False):
            input, target = self.mover(minibatch)
            loss, pred = self(input, target)
            self._invoke_metrics(loss, pred, target, Mode.EVAL)

    def update(self, *minibatch) -> None:
        '''Perform a single learning step. This method is normally invoked by
           the train method but can also be invoked directly. If there are
           metrics configured, they will be invoked and the result is returned
           together with the loss value. The data will be moved to the
           device using the configured mover.

           Args:
               minibatch: the input and target tensors
        '''
        self.model.train()
        with torch.set_grad_enabled(True):
            input, target = self.mover(minibatch)
            loss, pred = self(input, target)
            loss.backward()
            self.optim.step()
            self.step += 1
            self._invoke_metrics(loss, pred, target, Mode.TRAIN)
            self.optim.zero_grad()

    def stop(self) -> Type[NoReturn]:
        '''Will stop the training early. Typcially invoked by a callback when the
           training is not progressing anymore.'''
        raise self.StopError()

    def _invoke_callbacks(self, callbacks: List[Callable], phase: Mode) -> None:
        for callback in callbacks:
            callback(self, phase)

    def fit(self, data: Iterable, valid_data: Iterable = None,
            epochs: int = 1, callbacks: List[Callable] = None) -> None:
        '''Run the training and optionally the validation for a number of epochs.
           If no validation data is provided, the validation cycle is skipped.
           If the validation should not run every epoch, check the `Skipper`
           class.

           Args:
               data: the data to use for the training
               valid_data: the data to use for the validation, default = None.
               epochs (int): the number of epochs to run the training for,
               default = 1
               callbacks: the callbacks to use. These are invoked at the end of an update
               and the end of the validation. The default is the `PrintMeter` that will
               print an update at the end of each epoch and ignore the other updates.
        '''
        if callbacks is None:
            # pylint: disable=C0415, R0401
            from .callbacks import PrintMeter
            callbacks = [PrintMeter()]

        try:

            self.batches = len(data)

            for _ in range(epochs):
                self.epoch += 1

                for minibatch in data:
                    self.update(*minibatch)
                    self._invoke_callbacks(callbacks, Mode.TRAIN)

                if valid_data is not None:
                    for minibatch in valid_data:
                        self.validate(*minibatch)

                self.update_history("epoch", self.epoch)
                self._invoke_callbacks(callbacks, Mode.EVAL)

        except self.StopError:
            pass

    def save(self, filename: str = None) -> str:
        '''Save the training state to a file. This includes the underlying model state
           but also the optimizer state and internal state. This makes it
           possible to continue training where it was left off.

           Please note::
               This method doesn't store the model itself, just the trained parameters.
               It is recommended to use regular version control like `git` to save
               different versions of the code that creates the model.

           If no filename is provide, a directory and filename will be generated using
           the following pattern:

                   `./models/[workout.id]/workout_[model.step].pty`

           Args:
               filename (str): the name of the file to store the training state.
        '''

        if filename is None:
            subdir = "./models/{}/".format(self._id)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            filename = "{}workout_{:08d}.pty".format(subdir, self.step)


        state = {
            "step": self.step,
            "id": self._id,
            "model": self.model.state_dict(),
            "epoch": self.epoch,
            "history": self.history,
            "optim": self.optim.state_dict()
        }

        torch.save(state, filename)
        return filename

    def load(self, filename: str = None) -> str:
        '''Restore previously stored workout.

           If no filename is provided it will try to find the last stored training
           file and will use that one. The algoritm assumed that directories
           and files can be sorted based on its name to find the latest version. This is
           true is you use the let Fos determine the filename, but might not be the case
           if you provided your own filename during the `save` method.

           Args:
               filename (str): The filename of the training state to load.
        '''

        if filename is None:
            filename = _find_latest_training("./models/")

        state = torch.load(filename)
        self._id = state["id"]
        self.step = state["step"]
        self.epoch = state["epoch"]
        self.history = state["history"]
        self.model.load_state_dict(state["model"])
        self.optim.load_state_dict(state["optim"])
        return filename


def _find_latest_training(rootdir: str) -> str:
    '''Find the last saved training file. It will using sorting to determine
       what is the lastest training file. Expected structure:

       rootdir
            workoutid1
                trainingfile1
                trainingfile2
            workoutid2
                trainingfile1
            workoutid3
                trainingfile1
                trainingfile2
                trainingfile3 # This file will be selected

       Args:
           rootdir (str): The root directory where to start the search
    '''
    try:
        subdir = sorted(os.listdir(rootdir))[-1]
        filename = sorted(os.listdir(rootdir + subdir))[-1]
        return os.path.join(rootdir, subdir, filename)
    except OSError:
        logging.warning(
            "Couldn't find previously saved training files at directory %s",
            rootdir)
        return None


class AMPWorkout(Workout):
    """"Workout to uses the Automatic Mixed Precision (AMP) feature. This feature enables
    automatic conversion of certain GPU operations from FP32 precision to mixed precision,
    thus improving performance while maintaining accuracy."""

    # pylint: disable= R0902
    def __init__(self, model: nn.Module, loss_fn: Callable,
                 optim: Optimizer = None, mover=None, **metrics):

        if not torch.cuda.is_available():
            raise Exception("AMP based training requires CUDA support")

        super().__init__(model, loss_fn, optim, mover, **metrics)
        self.scaler = torch.cuda.amp.GradScaler()

    def update(self, *minibatch) -> None:
        self.model.train()
        scaler = self.scaler
        with torch.set_grad_enabled(True):
            input, target = self.mover(minibatch)

            with torch.cuda.amp.autocast():
                loss, pred = self(input, target)

            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()

            self.step += 1
            self._invoke_metrics(loss, pred, target, Mode.TRAIN)
            self.optim.zero_grad()


# pylint: disable=too-many-ancestors
class SmartHistory(dict):
    '''Stores the values of a metric. In essence it is a dictionary with the
    key being the step when the metric was calculated and the value being the
    outcome of that calculation.

    If multiple values per step are received (like is the case with validation)
    the moving average of the metric values are stored instead. So at every step
    there is only a single value per metric.

    Finally it will delay calling `value.item` by putting received values on a
    backlog. This will typically result in less blocking behavior.

    Args:
        momentum: The momentum to use for the moving average (default = 0.9). The
        calculation used is: momentum*old + (1-momentum)*new
    '''

    def __init__(self, momentum=0.9, max_backlog=10):
        super().__init__()
        self.momentum = momentum
        self._backlog = []
        self.max_backlog = max_backlog

    def _process_backlog(self):
        for step, value in self._backlog:
            if hasattr(value, "item"):
                value = value.item()
            if super().__contains__(step):
                old_value = super().__getitem__(step)
                value = self.momentum * old_value + (1-self.momentum) * value
            super().__setitem__(step, value)
        self._backlog = []

    def __setitem__(self, step: int, value):
        self._backlog.append((step, value))
        if len(self._backlog) > self.max_backlog:
            self._process_backlog()

    def __getitem__(self, step: int):
        if self._backlog:
            self._process_backlog()
        return super().__getitem__(step)

    def __contains__(self, step: int):
        if self._backlog:
            self._process_backlog()
        return super().__contains__(step)


class Mover():
    '''Moves tensors to a specific device. This is used to move
       the input and target tensors to the correct device like a GPU. Normally
       the default mover will be fine and you don't have to specify one
       explictely when you create the Workout.

       Args:
           device: The device to move the tensors to
           non_blocking: Use a non-blocking operation (asynchronous move), default = True

       Example usage:

       .. code-block:: python

           mover    = Mover("cuda", non_blocking=False)
           trainer  = Workout(..., mover=mover)
    '''

    def __init__(self, device, non_blocking: bool = True):
        self.device = device
        self.non_blocking = non_blocking

    @staticmethod
    def get_default(model: nn.Module):
        '''Get a mover based on the device on which the parameters of
           the model resides. This method is also called by the workout if
           there is no mover provided as an argument when creating a new workout.
        '''
        device = next(model.parameters()).device
        return Mover(device)

    def __call__(self, batch):
        '''Move a minibatch to the correct device'''

        if torch.is_tensor(batch):
            return batch.to(device=self.device, non_blocking=self.non_blocking)

        if isinstance(batch, (list, tuple)):
            # batch = [self(row) for row in batch]
            return tuple(self(elem) for elem in batch)

        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch) # pylint: disable=E1101
            return batch.to(device=self.device, non_blocking=self.non_blocking)

        logging.warning(
            "This mover doesn't support batch elements of type %s",
            type(batch))
        return batch
