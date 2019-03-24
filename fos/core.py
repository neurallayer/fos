'''
The core module contains the basic functionality required to train a model. The only additional
functionality you would normally include in your application is a `Meter` that will
display the progress during the training.

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
from fos.metrics import LossMetric


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
           mover: the mover to use. If None is specified, a default mover will be created to move
           tensors to the correct device


       Example usage:

       .. code-block:: python

           model = Supervisor(preditor, F.mse_loss, {"acc": BinaryAccuracy()})
    '''

    def __init__(self, predictor, loss_fn, metrics=None, mover=None):
        super().__init__()
        self.predictor = predictor
        self.metrics = metrics if metrics is not None else []
        self.loss_fn = loss_fn
        self.loss_metric = LossMetric()
        self.step = 0
        self.mover = mover if mover is not None else Mover.get_default(predictor)

    def update_metrics(self, loss, input, target):
        '''Invoke the configured metrics functions and return the result

           Args:
               loss (scaler): the loss value
               input (Tensor): the predicted value
               target (Tensor): the target value
        '''
        self.loss_metric.update(loss, None)
        for metric in self.metrics:
            metric.update(input, target)

    def get_metrics(self):
        '''Get all the configured metrics'''
        return [self.loss_metric] + self.metrics
            
    def reset_metrics(self):
        for metric in self.get_metrics():
            metric.reset()

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
        '''Predict a batch of data at once and return the result. No metrics
           will be generated when predicting values. The data will be moved to the
           device using the configured mover.

           Args:
               input (Tensor): the batch of input tensors
        '''
        self.eval()
        with torch.set_grad_enabled(False):
            input = self.mover(input)
            pred = self.predictor(input)
            return pred
        
    def validate(self, input, target):
        '''Perform a single validation iteration. If there are metrics
           configured, they will be invoked and the result is returned together
           with the loss value. The data will be moved to the
           device using the configured mover.

           Args:
               input (Tensor): the input tensor
               target (Tensor): the target tensor
        '''
        self.eval()
        with torch.set_grad_enabled(False):
            input, target = self.mover(input), self.mover(target)
            loss, pred = self(input, target)
            return self.update_metrics(loss, pred, target)

    def learn(self, input, target):
        '''Perform a single learning step. This method is normally invoked by
           the trainer but can also be invoked directly. If there are metrics
           configured, they will be invoked and the result is returned together
           with the loss value. The data will be moved to the
           device using the configured mover.

           Args:
               input (Tensor): the input data
               target (Tensor): the target data
        '''
        self.train()
        with torch.set_grad_enabled(True):
            self.step += 1
            input, target = self.mover(input), self.mover(target)
            loss, pred = self(input, target)
            loss.backward()
            return self.update_metrics(loss, pred, target)

    def state_dict(self):
        return {
            "step": self.step,
            "predictor": self.predictor.state_dict()
        }

    def load_state_dict(self, state):
        self.step = state["step"]
        self.predictor.load_state_dict(state["predictor"])


class Trainer():
    '''Train your supervised model. After creating an instance of a Trainer,
       you can start the training the model by invoking the `run` method.

       Args:
           model (Supervisor): the supervised model that needs to be trained
           optim (Optimizer): the optimizer to use to update the model
           meter (Meter): what meter should be used to handle and display metrics
           metrics (dict): The model metrics (like gradients) that should be generated
            during training (model metrics are not applied during the validation phase)

       Example usage:

       .. code-block:: python

           trainer = Trainer(model, optim, meter)
           trainer.run(data, epochs=10)
           trainer.save()

    '''

    def __init__(self, model, optim, meter, metrics=None):
        self.model = model
        self.optim = optim
        self.metrics = metrics if metrics is not None else []
        self.epoch = 0
        self.id = str(int(time.time()))
        self.meter = meter

    def _display_meter(self, phase, progress=None):
        ctx = {
            "step": self.model.step,
            "epoch": self.epoch,
            "phase": phase,
            "progress": progress
        }
        metrics = self.model.get_metrics() + self.metrics
        self.meter.display(metrics, ctx)

        
    def update_metrics(self):
        '''call the configured metrics and update them
           accordingly.
        '''
        for metric in self.metrics:
            metric.update(self.model, self.optim)

    def _update_model(self):
        '''Update the model. At this point the model has performed both the
        forward and backward step. The default implementation invokes the
        `optimizer.step()` to perform the updates and afterwards reset the
        gradients with `optimizer.zero_grad()`.
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
        steps_per_epoch = len(data)
        self.model.reset_metrics()
        for idx, (input, target) in enumerate(data):
            self.model.learn(input, target)
            self.update_metrics()
            self._update_model()
            progress = (1. + idx) / steps_per_epoch
            self._display_meter("train", progress)

    def validate(self, data):
        '''Validate the model using the data provided. Typically you
           would use `trainer.run` instead since that will also handle lifecycle
           of meters, unless you only want to run a validation cycle.

           Args:
               data: the validation data to use.
        '''
        self.model.reset_metrics()
        for input, target in data:
            self.model.validate(input, target)
        self._display_meter("valid")

        
    def run(self, data, valid_data=None, epochs=1):
        '''Run the training and optionally the validation for a number of epochs.
           If no validation data is provided, the validation cycle is skipped. If
           the validaiton should not run every epoch, check the `Skipper` class.

           Args:
               data: the data to use for the training
               valid_data: the data to use for the validation, default = None.
               epochs (int): the number of epochs to run the training for, default = 1
        '''
        try:
            for _ in range(epochs):
                self.meter.reset()
                self.train(data)
                if valid_data is not None:
                    self.validate(valid_data)
                self.epoch += 1
        finally:
            self.meter.reset()

            
    def predict(self, data, ignore_label=False, auto_flatten=True):
        '''Predict the outcome given the provided input data. Data is expected to be an iterable, 
           typically a PyTorch dataloader. 
           
           Args:
               data: the input data to use
               ingore_label: returns the dataloader a label/target value that should be ignored. This enables to 
               use a dataloader that returns both the X and y values. 
               auto_flatten: should the last dimension be flatten if that dimension is 1.

           Note: All results are stored in memory. This can cause memory issues if you have a lot of data and 
           large prediction tensors.
        '''
        result = None
        for batch in data:
            
            if ignore_label:
                batch = batch[0]

            preds = self.model.predict(batch)

            if torch.is_tensor(preds):
                preds  = (preds,)

            if result is None:
                if not isinstance(preds, tuple):
                    raise ValueError("Cannot handle result type ", type(preds))
                else:
                    result = [[] for _ in preds]

            for r,p in zip(result, preds):
                if auto_flatten and (p.shape[-1] == 1):
                    p = p.flatten(-1)
                r.extend(p.cpu().numpy())

        if len(result) == 1:
            return np.array(result[0])
        else:
            return np.array(result)

            
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
               This method doesn't store the model itself, just the trained parameters.
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
        return filename


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
        return filename


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
       the default mover will be fine and you don't have to specify one
       explictely when you create the Supervisor.

       Args:
           device: The device to move the tensors to
           non_blocking: Use a non-blocking operation (asynchronous move), default = True

       Example usage:

       .. code-block:: python

           mover    = Mover("cuda", non_blocking=False)
           trainer  = Supervisor(..., mover=mover)
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

    def __call__(self, batch):
        '''Move a single batch to the correct device'''

        if torch.is_tensor(batch):
            return batch.to(device=self.device, non_blocking=self.non_blocking)

        if isinstance(batch, (list, tuple)):
            batch = [self(row) for row in batch]
            return batch

        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
            return batch.to(device=self.device, non_blocking=self.non_blocking)

        logging.warning(
            "This mover doesn't support batch of type %s",
            type(batch))
        return batch
