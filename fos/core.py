'''
The core module contains the basic functionality required to train a model. The only additional
functionality you would normally include in your application is a `Meter` that will
display the progress during the training.

There are 3 classes in the core package:

1) Supervisor, that creates a supervised model with a loss function
2) Trainer, that will run the training including validation
3) Mover, that will move tensors to a device like a GPU

'''

import time
import os
import numpy as np
import torch
import torch.nn as nn
import logging


class Supervisor(nn.Module):
    '''Supervisor extends a regular PyTorch model with a loss function. This class extends from 
       `nn.Module` but adds methods that the trainer will use to train and validate the model. Altough
       not typical, it is possible to have multiple trainers train the same model.
       
       Besides the added loss function, also additional metrics can be specified to get insights into
       the performance of model. Normally you won't call a supervisor directly but rather interact with 
       an instance of the Trainer_ class.

       Args:
           predictor (nn.Module): The model that needs to be trained.
           loss_fn (function): The loss function (objective) that should be used to train the model
           metrics (dict): The metrics that should be evaluated during the training and validation
           
       Example usage:
       
       .. code-block:: python

           model = Supervisor(preditor, F.mse_loss, {"acc": BinaryAccuracy()})
    '''

    def __init__(self, predictor, loss_fn, metrics=None):
        super().__init__()
        self.predictor = predictor
        self.metrics = metrics if metrics is not None else {}
        self.loss_fn = loss_fn
        self.step = 0

    def handle_metrics(self, loss, input, target):
        '''Invoke the configured metrics functions and return the result

           Args:
               loss (scaler): the loss value
               y (Tensor): the predicted value
               target (Tensor): the target value
        '''
        output = {}

        output["loss"] = loss
        for key, fn in self.metrics.items():
            value = fn(input, target)
            if value is not None:
                output[key] = value
        return output

    def forward(self, input, target):
        '''Implementation of the forward method in nn.Module

           Args:
               input (Tensor): the input data for the model
               target (Tensor): the target data for the loss function
        '''
        pred = self.predictor(input)
        loss = self.loss_fn(pred, target)
        return loss, pred

    def predict(self, input):
        '''Predict a single batch of data and return the result. No metrics
           will be generated when predicitng values.

           Args:
               input (Tensor): the input tensor
        '''
        pred = self.predictor(input)
        return pred

    def validate(self, input, target):
        '''Perform a single validation iteration. If there are metrics
           configured, they will be invoked and the result is returned together
           with the loss value.

           Args:
               input (Tensor): the input tensor
               target (Tensor): the target tensor
        '''
        loss, pred = self(input, target)
        return self.handle_metrics(loss.item(), pred, target)

    def learn(self, input, target):
        '''Perform a single learning step. This method is normally invoked by
           the trainer but can also be invoked directly. If there are metrics
           configured, they will be invoked and the result is returned together
           with the loss value.

           Args:
               input (Tensor): the input data
               target (Tensor): the target data
               optim (Optimizer): the optimizer to use to update the model

        '''
        self.step += 1
        loss, pred = self(input, target)
        loss.backward()
        return self.handle_metrics(loss.item(), pred, target)

    def state_dict(self):
        return {
            "step": self.step,
            "predictor": self.predictor.state_dict()
        }

    def load_state_dict(self, state):
        self.step = state["step"]
        self.predictor.load_state_dict(state["predictor"])


class Trainer():
    '''Train your supervised model using an optimizer.

       Args:
           model (Supervisor): the supervised model that needs to be trained
           optim (Optimizer): the optimizer to use to update the model
           meter (Meter): what meter should be used to handle and display metrics
           metrics (dict): The model metrics (like gradients) that should be generated
            during training (model metrics are not applied during the validation phase)
           mover: the mover to use. If None is specified, a default mover will be created to move
            tensors to the correct device
               
       Example usage:

       .. code-block:: python

           trainer = Trainer(model, optim, meter)
           trainer.run(data, epochs=10)
           trainer.save()

    '''

    def __init__(self, model, optim, meter, metrics=None, mover=None):
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

    def _update_model(self):
        '''Update the model. At this point the model has performed both the
        forward and backward step. The default implementation invokes the
        `optimizer.step` to perform the updates and afterwards reset the
        gradients.
        '''
        self.optim.step()
        self.optim.zero_grad()

    def train(self, data):
        '''Train the model using the training data provided. Typically you
           would use `trainer.run` instead since that will also handle lifecycle
           of meters.

           Args:
               data: the training data to use.
        '''
        self.model.train()
        with torch.set_grad_enabled(True):
            steps_per_epoch = len(data)
            for idx, (input, target) in enumerate(self.mover(data)):
                output = self.model.learn(input, target)
                self._update_meter(output)
                self._handle_metrics()
                self._update_model()
                progress = (1. + idx) / steps_per_epoch
                self._display_meter("train", progress)

    def validate(self, data):
        '''Validate the model using the data provided. Typically you
           would use `trainer.run` instead since that will also handle lifecycle
           of meters, unless you only want to run a vilidation cycle.

           Args:
               data: the validation data to use.
        '''

        self.model.eval()
        with torch.set_grad_enabled(False):
            for input, target in self.mover(data):
                output = self.model.validate(input, target)
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
            for input in self.mover(data):
                pred = self.model.predict(input)
                result.extend(pred.cpu().detach().numpy())

        return np.array(result)

    def run(self, data, valid_data=None, epochs=1):
        '''Run the training and optionally the validation for a number of epochs.
           If no validation data is provided, the validation cycle is skipped. If
           the validaiton should not run every epoch, check the `Skipper` class.

           Args:
               data: the data to use for the training
               valid_data: the data to use for the validation, default = None.
               epochs (int): the number of epochs to run the training for, default = 1
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
           continue training where it was left off.

           Please note::
               This method doesn't store the model itself, just the trained paramters.
               It is recommended to use regular version control like `git` to save
               different versions of the code that creates the model.

           If no filename is provide, a directory and filename will be generated using
           the following pattern:

                   `./models/[trainer.id]/trainer_[model.step].pty`

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
        '''Restore previously stored training program.

           If no filename is provided it will try to find the last stored training
           file and will use that one. The algoritm assumed that directories
           and files can be sorted based on its name to find the latest version. This is
           true is you use the let Fos determine the filename, but might not be the case
           if you provided your own filename during the `trainer.save` method.

           Args:
               filename (str): The filename of the training state to load.
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
        logging.warning(
            "Couldn't find previously saved training files at directory %s",
            rootdir)
        return None


class Mover():
    '''Moves tensors to a specific device. This is used to move
       the input and target tensors to the correct device. Normally
       the default mover will be fine and you don't have to specify one when
       you create the Trainer.

       Args:
           device: The device to move the tensors to
           non_blocking: Use a non-blocking operation (asynchronous move), default = True

       Example usage:
       
       .. code-block:: python

           mover    = Mover("cuda")
           trainer  = Trainer(..., mover=mover)
    '''

    def __init__(self, device, non_blocking=True):
        self.device = device
        self.non_blocking = non_blocking

    @staticmethod
    def get_default(model):
        '''Get a mover based on the device on which the parameters of
           the model resides. This method is also called by the trainer if
           there is no mover provided as an argument when creating a new trainer
        '''
        device = next(model.parameters()).device
        return Mover(device)

    def move(self, batch):
        '''Move a single batch to the device'''

        if torch.is_tensor(batch):
            return batch.to(device=self.device, non_blocking=self.non_blocking)

        if isinstance(batch, (tuple, list)):
            return tuple([self.move(elem) for elem in batch])

        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
            return batch.to(device=self.device, non_blocking=self.non_blocking)

        logging.warning(
            "This mover doesn't support batch of type %s",
            type(batch))
        return batch

    def __call__(self, data):
        for batch in data:
            yield self.move(batch)
