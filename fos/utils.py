"""
Collection of utilities that come in handy when training a model. Typially these can be used
when optimizing training with more advanced concepts like freezing layers or
annealing learning rates.
"""
import os
import random
from typing import Iterable
import torch
import numpy as np


class ScalableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. User can specify ``num_samples`` to draw.

    This sampler handles large datasets better then the default RandomSampler that
    comes with PyTorch but is more restricted in functionality. Samples are always drawn
    with replacement (but that is typically less of an issue with large datasets).

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        low_mem (bool): if memory is sparse use this option to avoid
        allocating additional memory

    Example usage:

       .. code-block:: python

            sampler = ScalableRandomSampler(dataset, num_samples=10000)
            data_loader = Dataloader(dataset, sampler=sampler, ...)
    """

    def __init__(self, data_source, num_samples=None, low_mem=False):
        # don't call super since it is a no-op
        super().__init__(None)
        self.data_source = data_source
        self.num_samples = num_samples
        self.low_mem = low_mem

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))

    def __iter__(self):
        max_idx = len(self.data_source)
        if self.low_mem:
            # Doesn't allocate a temporary array
            for _ in range(self.num_samples):
                yield np.random.randint(0, max_idx)
        else:
            # This is a faster method but creates a temporary array
            for idx in np.random.randint(0, max_idx, self.num_samples):
                yield idx

    def __len__(self):
        return self.num_samples

# pylint: disable=C0111
class SmartOptimizer():
    '''Add clipping and scheduling capabilities to a regular optimizer.

    Args:
        optim (Optimizer): the optimizer to use
        clipper (tuple): clipping parameters as a tuple (max_norm, norm_type). See also
            `torch.nn.utils.clip_grad_norm_` for more details
        scheduler (Scheduler): the scheduler to use

    Example usage:

    .. code-block:: python

        smart_optim = SmartOptimizer(optim, clipper=(1,2), scheduler=my_scheduler)
    '''

    def __init__(self, optim, clipper=None, scheduler=None):
        self.optim = optim
        self.clipper = clipper
        self.scheduler = scheduler

    def _clip(self):
        max_norm, norm_type = self.clipper
        for group in self.optim.param_groups:
            torch.nn.utils.clip_grad_norm_(
                group["params"], max_norm=max_norm, norm_type=norm_type)

    # pylint: disable=W0613
    def step(self, closure=None):
        if self.clipper is not None:
            self._clip()
        self.optim.step()
        if self.scheduler is not None:
            self.scheduler.step()

    @property
    def param_group(self):
        """Get the param group of the underlying optimizer"""
        return self.optim.param_group

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def zero_grad(self):
        self.optim.zero_grad()

    def state_dict(self):
        return {
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None
        }

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict["optim"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])


class Skipper():
    '''Wrap a dataloader and skips epochs. Typically used when you don't
    want to run the validation at every epoch.

    Arguments:
        dl: the dataloader (or any other iterable) that needs to be wrapped
        skip (int): how many epochs should be skipped. If skips is for example 3
        the iterator is only run at every third epoch.

    Example usage:

    .. code-block:: python

        # Run the validation only every 5th epoch
        valid_data = Skipper(valid_data, 5)
        trainer.run(train_data, valid_data, 20)
    '''

    def __init__(self, dl, skip):
        self._dl = dl
        self.skip = skip
        self.cnt = 1

    def __len__(self):
        i = self._get_iter()
        if hasattr(i, "__len__"):
            return i.__len__()
        return 0

    def _get_iter(self):
        return self._dl.__iter__() if ((self.cnt % self.skip) == 0) else iter([])

    def __iter__(self):
        i = self._get_iter()
        self.cnt += 1
        return i

def init_random(seed=0, cudnn=False):
    '''Initialize the random seeds for `python`, `torch` and `numpy` in order to improve
    reproducability. This makes for example the initialization of
    weights in the different layers of the model reproducable.

     Example usage:

     .. code-block:: python

        init_random()

    Args:
        seed (int): the seed to use. default = 0
        cudnn (bool): should we also disable some of the smart (non deterministic)
        optimimalizations of CuDNN. This might impact performance, so only recommended if
        really required. default = False
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_params(model):
    '''Print an overview of the model parameters, their status and their names. When you want
       to freeze parameters, this will help you to indentify their names.
    '''
    for idx, (name, layer) in enumerate(model.named_parameters()):
        text = "[unfrozen]" if layer.requires_grad else "[frozen]"
        print("{:3} {:10} {:50} {}".format(idx, text, name, tuple(layer.shape)))

def _set_requires_grad(model, req_grad, param_name):
    for name, param in model.named_parameters():
        if name.startswith(param_name):
            param.requires_grad = req_grad

def freeze(model, param_name=""):
    '''Freeze model parameters based on their name. If no name is provided, it will freeze
        all the parameters. See also print_params.

        Args:
            param_name (str): The first part of a parameter name to freeze. Can be a single string or
            a set of strings.
    '''
    _set_requires_grad(model, False, param_name)

def unfreeze(model, param_name=""):
    '''Unfreeze a number of parameters based on their name. If no name is provided, it will
        unfreeze all the parameters. See also print_params.

        Args:
            layerparam_name_name (str): The first part of the parameter name to unfreeze. Can be a single
            string or a set of strings.
    '''
    _set_requires_grad(model, True, param_name)




def get_normalization(dataloader: Iterable, max_iter=None, feature_dim=1):
    '''Calculates the mean and standard deviation for the data from a
    dataloader. The output can be used for normalizing the images
    before feeding them to a neural network.

    The dataloader should return either just the batch of tensors or
    a tuple/list of which the first element is the tensor.

    Args:
        dataloader: The datalaoder you want to use to get the images
        max_iter: Limit the number of batches to consider. If None, all the iterations in the
         dataloader will be used. It should be noted that if a batch has more then one sample,
         the actual number of samples equals: samples = max_iter * batch_size
         feature_dim: which dimension has the features to normalize.

    Example usage:

    .. code-block:: python

        # Image tensors with the format  NxCxWxH (PyTorch format)
        n = get_normalization(image_loader, 1000, 1)
    '''

    std = 0.
    mean = 0.
    step = 0
    first = True
    axis = []

    for data in dataloader:
        step += 1

        if max_iter is not None:
            if step > max_iter:
                break

        if not torch.is_tensor(data):
            data = data[0]

        data = data.cpu().numpy()

        if first:
            axis = list(range(len(data.shape)))
            del axis[feature_dim]
            axis = tuple(axis)
            first = False

        std += np.std(data, axis=axis)
        mean += np.mean(data, axis=axis)

    return {"mean": mean / step, "std": std / step}
