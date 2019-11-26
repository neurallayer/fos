'''
The core module contains the basic functionality required to train a model.
The only additional functionality you would normally include in your
application is a `Meter` that will display the progress during the training.

There are 3 classes in the core package:

1) Supervisor, that creates a supervised model with a loss function
2) Trainer, that will run the training including the validation
3) Mover, that will move tensors to a device like a GPU

'''

import time
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from .meters import PrintMeter
from torch.jit import trace
from enum import Enum
from typing import Callable


class Phase(Enum):
    TRAIN = 1
    VALID = 2
    OTHER = 3


class StopError(Exception):
    '''used internally to stop the training before all the epochs have finished'''
    pass


class SmartHistory(dict):
    '''Stores the values of a metric. In essence it is a dictionary with the
    key being the step when the metric was calculated and the value being the
    outcome of that calculation.

    If multiple values per step are received (like is the case with validation)
    the moving average of the metric values are stored instead.

    Args:
        momentum: The momentum to use for the moving average (default = 0.9). The
        calculation used is: momentum*old + (1-momentum)*new
    '''

    def __init__(self, momentum=0.9):
        super().__init__()
        self.momentum = momentum

    def __setitem__(self, step: int, value):
        if step in self:
            value = self.momentum * self[step] + (1-self.momentum) * value
        super().__setitem__(step, value)


def empty_cb(*args):
    pass


class Workout(nn.Module):
    '''Coordinates all the training of a model Workout and provides many methods that
       reduces the amount of boilerplate code when training a model. In its the simplest form:

       .. code-block:: python

           workout = Workout(mymodel, F.mse_loss)
           workout.fit(data, epochs=10)

       Besides the added loss function and optional optimizer, also additional metrics can be specified
       to get insights into the performance of model.

       Args:
           model (nn.Module): The model that needs to be trained.
           loss_fn (function): The loss function (objective) that should be used to train the model
           optim: The optimizer to use. If none is provided, SGD will be used
           mover: the mover to use. If None is specified, a default mover will be created to move
           tensors to the correct device.
           metrics : The metrics that should be evaluated during the training and validation. Every metric
           can be specified as an optional argument, e.g acc=BinaryAccuracy()

       Example usage:

       .. code-block:: python

           workout = Workout(model, F.mse_loss, Adam(model.paramters()), acc=BinaryAccuracy())
    '''

    def __init__(self, model: nn.Module, loss_fn: Callable, optim=None, mover=None, **metrics):
        super().__init__()
        self.model = model
        self.metrics = metrics
        self.loss_fn = loss_fn
        self.mover = mover if mover is not None else Mover.get_default(model)
        self.history = {}
        self.step = 0
        self.epoch = 0
        self.batches = None
        self.id = str(int(time.time()))
        self.optim = optim if optim is not None else torch.optim.SGD(
                                                model.parameters(), lr=1e-3)

    def update_history(self, name: str, value):
        ''''Update the history for the passed metric name and value. It will store
            store the metric under the current step.
        '''
        if name not in self.history:
            self.history[name] = SmartHistory()
        self.history[name][self.step] = value

    def get_metricname(self, name: str, phase: str):
        '''Get the fully qualified name for a metric. If phase equals train the
           metric name is as specified and if phase is "valid" the metric name
           is "val_" + name.

           So for example wehn performing a trianing cycle with validaiton step
           the following two loss metrics will be availble:

           "loss" : for the recorded loss during training
           "val_loss": or the recorded loss during validation
        '''
        return name if phase == "train" else "val_" + name

    def update_metrics(self, loss, pred, target, phase: str):
        '''Invoke the configured metrics functions and return the result

           Args:
               loss (scaler): the loss value
               pred (Tensor): the predicted value
               target (Tensor): the target value
        '''
        loss_name = self.get_metricname("loss", phase)
        self.update_history(loss_name, loss.item())
        for name, fn in self.metrics.items():
            value = fn(pred, target)
            fqname = self.get_metricname(name, phase)
            self.update_history(fqname, value)

    def get_metrics(self):
        '''Get all metrics that have at least one value logged'''
        return self.history.keys()

    def has_metric(self, name: str, step=None):
        '''Check for the metric value for the provided metric name and step.
           True of it exist, False otherwise.
        '''
        if name not in self.history:
            return False
        step = step if step is not None else self.step
        return step in self.history[name]

    def get_metric(self, name: str, step=None):
        '''Get the metric value for the provided metric name and step.
            If no step is provided, the last workout step is used.
        '''
        step = step if step is not None else self.step
        return self.history[name][step]

    def forward(self, input, target):
        '''Implementation of the forward method in nn.Module

           Args:
               input (Tensor): the input data for the model
               target (Tensor): the target data for the loss function
        '''
        pred = self.model(input)
        loss = self.loss_fn(pred, target)
        return loss, pred

    def trace(self, input):
        '''Create a traced model and return it.

           Args:
               input (Tensor): a minibatch of input tensors
        '''
        self.model.train()
        with torch.set_grad_enabled(False):
            input = self.mover(input)
            traced_model = trace(self.model, example_inputs=input,
                                 check_trace=False)
            return traced_model

    def predict(self, input):
        '''Predict a batch of data at once and return the result. No metrics
           will be generated when predicting values. The data will be moved to
           the device using the configured mover.

           Args:
               input (Tensor): the minibatch of only input tensors
        '''
        self.model.eval()
        with torch.set_grad_enabled(False):
            input = self.mover(input)
            pred = self.model(input)
            return pred

    def validate(self, minibatch: tuple):
        '''Perform a single validation iteration. If there are metrics
           configured, they will be invoked and the result is returned together
           with the loss value. The data will be moved to the
           device using the configured mover.

           Args:
               minibatch: the input and target tensor as a tuple
        '''
        self.model.eval()
        with torch.set_grad_enabled(False):
            input, target = self.mover(minibatch)
            loss, pred = self(input, target)
            return self.update_metrics(loss, pred, target, "valid")

    def update(self, minibatch: tuple):
        '''Perform a single learning step. This method is normally invoked by
           the train method but can also be invoked directly. If there are
           metrics configured, they will be invoked and the result is returned
           together with the loss value. The data will be moved to the
           device using the configured mover.

           Args:
               minibatch: the input and target tensors as a tuple
        '''
        self.model.train()
        with torch.set_grad_enabled(True):
            input, target = self.mover(minibatch)
            loss, pred = self(input, target)
            loss.backward()
            self.optim.step()
            self.step += 1
            self.update_metrics(loss, pred, target, "train")
            self.optim.zero_grad()

    def stop(self):
        '''Will stop the training early. Typcially invoked by a callback when the
        training is not progressing anymore.'''
        raise StopError()

    def state_dict(self):
        return {
            "step": self.step,
            "id": self.id,
            "model": self.model.state_dict(),
            "epoch": self.epoch,
            "history": self.history,
            "optim": self.optim.state_dict()
        }

    def load_state_dict(self, state: dict):
        self.id = state["id"]
        self.step = state["step"]
        self.epoch = state["epoch"]
        self.history = state["history"]
        self.model.load_state_dict(state["model"])
        self.optim.load_state_dict(state["optim"])

    def fit(self, data, valid_data=None, epochs=1, cb=PrintMeter()):
        '''Run the training and optionally the validation for a number of epochs.
           If no validation data is provided, the validation cycle is skipped.
           If the validation should not run every epoch, check the `Skipper`
           class.

           Args:
               data: the data to use for the training
               valid_data: the data to use for the validation, default = None.
               epochs (int): the number of epochs to run the training for,
               default = 1
               cb: the callback to use. These are invoked at the end of an update
               and the end of the validation. The default is the PrintMeter that will
               print an update at the end of each epoch and ignore the other updates.
        '''
        try:

            self.batches = len(data)

            for _ in range(epochs):
                self.epoch += 1

                for minibatch in data:
                    self.update(minibatch)
                    cb(self, "train")

                if valid_data is not None:
                    for minibatch in valid_data:
                        self.validate(minibatch)

                self.update_history("epoch", self.epoch)
                cb(self, "valid")
        except StopError:
            pass

    def save(self, filename: str = None):
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
            subdir = "./models/{}/".format(self.id)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            filename = "{}workout_{:08d}.pty".format(subdir, self.step)
        torch.save(self.state_dict(), filename)
        return filename

    def load(self, filename: str = None):
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

        self.load_state_dict(torch.load(filename))
        return filename


def _find_latest_training(rootdir: str):
    '''Find the last saved training file.

       Args:
           rootdir (str): The root directory where to start the search
    '''
    try:
        subdir = sorted(os.listdir(rootdir))[-1]
        filename = sorted(os.listdir(rootdir + subdir))[-1]
        return os.path.join(rootdir, subdir, filename)
    except BaseException:
        logging.warning(
            "Couldn't find previously saved training files at directory %s",
            rootdir)
        return None


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

    def __init__(self, device, non_blocking=True):
        self.device = device
        self.non_blocking = non_blocking

    @staticmethod
    def get_default(model: nn.Module):
        '''Get a mover based on the device on which the parameters of
           the model resides. This method is also called by the workout if
           there is no mover provided as an argument when creating a new workout
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
            batch = torch.from_numpy(batch)
            return batch.to(device=self.device, non_blocking=self.non_blocking)

        logging.warning(
            "This mover doesn't support batch elements of type %s",
            type(batch))
        return batch
