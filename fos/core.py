import time
import os
import numpy as np
import torch
import torch.nn as nn
from .utils import Mover


class SuperModel(nn.Module):
    '''SuperModel is short for `SupervisedModel` and extends a regular model
       with a loss function. This class is still a valid `nn.Module` in PyTorch
       but adds functionality that the trainer will use to train the model.

       Args:
           predictor (nn.Module): The model that needs to be trained.
           loss_fn (function): The loss function (objective) that should be used to train the model
           metrics (dict): The metrics that should be generated during the training and validation
    '''

    def __init__(self, predictor, loss_fn, metrics=None):
        super().__init__()
        self.predictor = predictor
        self.metrics = metrics if metrics is not None else {}
        self.loss_fn = loss_fn
        self.step = 0

    def handle_metrics(self, loss, y, t):
        '''Invoke the configured metrics functions return the result

           Args:
               loss (scaler): the loss value
               y (Tensor): the predicted value
               t (Tensor): the target value
        '''
        output = {}

        output["loss"] = loss
        for key, fn in self.metrics.items():
            value = fn(y, t)
            if value is not None:
                output[key] = value
        return output

    def forward(self, x, t):
        '''Implementation of the forward method in nn.Module

           Args:
               x (Tensor): the input data for the model
               t (Tensor): the target data for the loss function
        '''
        y = self.predictor(x)
        loss = self.loss_fn(y, t)
        return loss, y

    def predict(self, x):
        '''Predict a single batch of data and return the result. No metrics
           will be generated when predicitng values.

           Args:
               x (Tensor): the input tensor
        '''
        y = self.predictor(x)
        return y

    def validate(self, x, t):
        '''Perform a single validation iteration. If there are metrics
           configured, they will be invoked and the result is returned together
           with the loss value.

           Args:
               x (Tensor): the input tensor
               t (Tensor): the target tensor
        '''
        loss, y = self(x, t)
        return self.handle_metrics(loss.item(), y, t)

    def learn(self, x, t):
        '''Perform a single learning step. This method is normally invoked by
           the trainer but can also be invoked directly. If there are metrics
           configured, they will be invoked and the result is returned together
           with the loss value.

           Args:
               x (Tensor): the input data
               t (Tensor): the target data
               optim (Optimizer): the optimizer to use to update the model

        '''
        self.step += 1
        loss, y = self(x, t)
        loss.backward()
        return self.handle_metrics(loss.item(), y, t)

    def state_dict(self):
        return {
            "step": self.step,
            "predictor": self.predictor.state_dict()
        }

    def load_state_dict(self, state):
        self.step = state["step"]
        self.predictor.load_state_dict(state["predictor"])


class Freezer():
    '''Provides functionality to freeze/unfreeze parameters in a model based
       on their name. This comes in most handy during transfer learning at
       the beginning of the training you only want to train the newly added layers.

       Args:
           model (nn.Module): the model you want to use.

       Examples:
           freezer = Freezer(my_Model)
           freezer.freeze() # freeze all layers
           freezer.unfreeze("fc") # unfreeze last layer
    '''

    def __init__(self, model):
        self.model = model

    def _get_params(self, layer_name=""):
        for name, param in self.model.named_parameters():
            if name.startswith(layer_name):
                yield param

    def _set_requires_grad(self, req_grad, layer_name=""):
        for param in self._get_params(layer_name):
            param.requires_grad = req_grad

    def freeze(self, layer_name=""):
        '''Freeze a number of layers based on their name. If no name is provided, it will freeze
           all layers.

           Args:
              layer_name (str): The first part of the layer_name. Can be a single string or a set of strings.
        '''

        self._set_requires_grad(False, layer_name)

    def unfreeze(self, layer_name=""):
        '''Unfreeze a number of layers based on their name. If no name is provided, it will unfreeze
           all layers.

           Args:
              layer_name (str): The first part of the layer_name. Can be a single string or a set of strings.
        '''

        self._set_requires_grad(True, layer_name)

    def summary(self):
        '''Print an overview of the parameters and their status.
        '''
        for idx, (name, layer) in enumerate(self.model.named_parameters()):
            text = "[unfrozen]" if layer.requires_grad else "[frozen]"
            print("{:3} {:10} {:50} {}".format(
                idx, text, name, tuple(layer.shape)))


class Trainer():
    '''Train your supervised model using an optimizer.

       Args:
           model (SuperModel): the supervised model that will be trained
           optim (Optimizer): the optimizer to use
           meter (Meter): what meter should be used to handle and diplsay the various metrics
           metrics (dict): The model metrics (like gradients) that should be generated
             during training (model metrics are not applied during the validation phase)
           mover: the mover to use. If None is specified, a default mover will be used to move tensors to
             the correct device
    '''

    def __init__(self, model, optim, meter=None, metrics=None, mover=None):
        self.model = model
        self.optim = optim
        self.metrics = metrics if metrics is not None else {}
        self.epoch = 0
        self.id = str(int(time.time()))
        self.mover = mover if mover is not None else Mover.get_default(model)
        self.meter = meter

    def _update_meter(self, output, prefix=""):
        for key, value in output.items():
            self.meter.update(prefix + key, value)

    def _display_meter(self, phase, progress=None):
        ctx = {
            "step": self.model.step,
            "epoch": self.epoch,
            "phase": phase,
            "progress": progress
        }
        self.meter.display(ctx)

    def _handle_metrics(self):
        '''call the configured metrics and update the meters
           accordingly.
        '''
        for key, fn in self.metrics.items():
            value = fn(self.model, self.optim)
            if value is not None:
                self.meter.update(key, value)

    def train(self, data):
        '''Train the model using the training data provided

           Args:
               data: the training data to use.
        '''

        self.model.train()
        with torch.set_grad_enabled(True):
            steps_per_epoch = len(data)
            for idx, (x, t) in enumerate(self.mover(data)):
                output = self.model.learn(x, t)
                self._update_meter(output)
                self._handle_metrics()
                self.optim.step()
                self.optim.zero_grad()
                progress = (1. + idx) / steps_per_epoch
                self._display_meter("train", progress)

    def validate(self, data):
        '''Validate/evaluate the model using the data provided

            Args:
               data: the validation data to use.
        '''

        self.model.eval()
        with torch.set_grad_enabled(False):
            for x, t in self.mover(data):
                output = self.model.validate(x, t)
                self._update_meter(output, "val_")
            self._display_meter("valid")

    def predict(self, data):
        '''Predict the outcome given the provided input data.
           Data is expected to be an iterable, like for example a
           dataloader or a numpy array.

           Args:
               data: the input data to use

           Note: since this method stores all the results before returning them,
           this is not well suited for large number of big result tensors due to
           memory usage.
        '''
        result = []

        self.model.eval()
        with torch.set_grad_enabled(False):
            for x in self.mover(data):
                y = self.model.predict(x)
                result.extend(y.cpu().detach().numpy())

        return np.array(result)

    def run(self, data, valid_data=None, epochs=1):
        '''Run the training and optionally the validation for a number of epochs.
           If no validation data is provided, the validation is skipped. If
           the validaiton should not run every epoch, check the Skipper class.

           Args:
               data: the data to use for the training
               valid_data: the data to use for the validation (default=None)
               epochs (int): the number of epochs to run the training for (default=1)
        '''
        for _ in range(epochs):
            self.meter.reset()
            self.train(data)
            if valid_data is not None:
                self.validate(valid_data)
            self.epoch += 1

        self.meter.reset()

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "id": self.id,
            "model": self.model.state_dict(),
            "meter": self.meter.state_dict(),
            "optim": self.optim.state_dict()
        }


    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.id = state["id"]
        self.model.load_state_dict(state["model"])
        self.meter.load_state_dict(state["meter"])
        self.optim.load_state_dict(state["optim"])


    def save(self, filename=None):
        '''Save the training state to a file. This includes the underlying model state
           but also the optimizer state and internal state. This makes it possible to
           continue training where left off.

           If no filename is provide, a directory and filename will be generated.

           Args:
               filename (str): the name of the file to store the training state.
        '''

        if filename is None:
            subdir = "./models/{}/".format(self.id)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            filename = "{}trainer_{:08d}.pty".format(subdir, self.model.step)
        torch.save(self.state_dict(), filename)

    def load(self, filename=None):
        '''Restore previously stored training program. If no filename is provided
           it will try to find the last stored training file and will use that one.

           Args:
               filename (str): The filename of th training programm to use.
        '''

        if filename is None:
            filename = _find_latest_training("./models/")

        self.load_state_dict(torch.load(filename))


def _find_latest_training(rootdir):
    '''Find the last saved training file.

       Args:
           rootdir (str): The root directory where to start the search
    '''
    try:
        subdir = sorted(os.listdir(rootdir))[-1]
        filename = sorted(os.listdir(rootdir + subdir))[-1]
        return os.path.join(rootdir, subdir, filename)
    except BaseException:
        print("Couldn't find previously saved training files at directory ", rootdir)
        return None
