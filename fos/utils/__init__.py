from abc import abstractmethod
import torch
import torchvision
import numpy as np
import logging


class BaseDataset(torch.utils.data.Dataset):
    '''Base Dataset that could be subclassed.
    '''

    def __init__(self, transform=None, transform_y=None):
        self.transform = transform
        self.transform_y = transform_y

    @abstractmethod
    def __len__(self):
        '''length of the dataset'''

    def __getitem__(self, idx):
        identifier = self.get_id(idx)

        input = self.get_x(identifier)
        if self.transform is not None:
            input = self.transform(input)

        target = self.get_y(identifier)
        if self.transform_y is not None:
            target = self.transform_y(target)

        return input, target

    def get_id(self, idx):
        '''default implementation returns the idx
           as identifier and doesn't perform any mapping.
        '''
        return idx

    @abstractmethod
    def get_x(self, identifier):
        '''Return the input value(s) for the model. In case there are multiple
           values, return a tuple.
        '''

    @abstractmethod
    def get_y(self, identifier):
        '''Return the target value(s) for the model.In case there are multiple
           values, return a tuple.'''


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
        for p in self.optim.param_groups:
            torch.nn.utils.clip_grad_norm_(
                p["params"], max_norm=max_norm, norm_type=norm_type)

    def step(self):
        if self.clipper is not None:
            self._clip()
        self.optim.step()
        if self.scheduler is not None:
            self.scheduler.step()

    @property
    def param_group(self):
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
    '''Wrap a dataloader and skip epochs. Typically used when you don't
    want to execute the validation every epoch.

    Arguments:
        dl: the dataloader (or another iterable) that needs to be wrapped
        skip (int): how many epochs should be skipped. If skips is for example 3
        the iterator is only run at every third epcoh.

    Example usage:
       
    .. code-block:: python

        # Run the validation only every 5th epoch
        valid_data = Skipper(valid_data, 5)
        trainer.run(train_data, valid_data, 20)
    '''

    def __init__(self, dl, skip):
        self.dl = dl
        self.skip = skip
        self.cnt = 1

    def __len__(self):
        i = self._get_iter()
        if hasattr(i, "__len__"):
            return i.__len__()
        return 0

    def _get_iter(self):
        return self.dl.__iter__() if ((self.cnt % self.skip) == 0) else iter([])

    def __iter__(self):
        i = self._get_iter()
        self.cnt += 1
        return i

    
def init_random(seed=0, cudnn=False):
    '''Initialize the random seeds for `torch` and `numpy` in order to improve 
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
    torch.manual_seed(0)
    np.random.seed(0)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
