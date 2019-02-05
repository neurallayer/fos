from abc import abstractmethod
import torch
import torchvision
import numpy as np


class Mover():
    '''Moves tensors to a specific device. This is used to move
       the input and target tensors to the same device as
       the model (often a GPU).

       Arguments:
           device: The device you want to move the tensors to
           non_blocking: Should it try to use a non-blocking operation (asynchronous move)

       Example:
           mover = Mover("cuda")
           for x,y in mover(dataloader):
           ...
    '''

    def __init__(self, device, non_blocking=True):
        self.device = device
        self.non_blocking = non_blocking

    @staticmethod
    def get_default(model):
        '''Get a mover based on the device on which the model resides.'''
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

        print("This mover doesn't support batch of type ", type(batch))
        print("Supported types: tuples, tensors and numpy arrays")
        return batch

    def __call__(self, data):
        for batch in data:
            yield self.move(batch)


def get_clipping_class(optimClass, max_norm=1., norm_type=2):

    class Clipper(optimClass):

        def step(self):
            for p in self.param_groups:
                torch.nn.utils.clip_grad_norm_(
                    p["params"], max_norm=max_norm, norm_type=norm_type)
            return super().step()

    return Clipper


def add_scheduler(optim, Scheduler):

    scheduler = Scheduler(optim)
    orig_step = optim.step

    def step():
        orig_step()
        scheduler.step()

    optim.step = step


class BaseDataset(torch.utils.data.Dataset):
    '''Base Dataset that should be subclassed.
    '''

    def __init__(self, transform=None, transform_y=None):
        self.transform = transform
        self.transform_y = transform_y

    @abstractmethod
    def __len__(self):
        '''length of the dataset'''

    def __getitem__(self, idx):
        i = self.get_id(idx)

        x = self.get_x(i)
        if self.transform is not None:
            x = self.transform(x)

        y = self.get_y(i)
        if self.transform_y is not None:
            y = self.transform_y(y)

        return x, y

    def get_id(self, idx):
        '''default implementation just return the idx
           as identifier.
        '''
        return idx

    @abstractmethod
    def get_x(self, identifier):
        '''Return the x value'''

    @abstractmethod
    def get_y(self, identifier):
        '''Return the target value'''


class ScalableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. User can specify ``num_samples`` to draw.

    This sampler handles large datasets better then the default RandomSampler
    but is more restricted in functionality. Samples are always drawn
    with replacement (but that is typically less of an issue with large datasets).

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        low_mem (bool): if memory is sparse use this option to avoid
        allocating additional memory
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
            # Doesn't allocate much additional memory
            for _ in range(self.num_samples):
                yield np.random.randint(0, max_idx)
        else:
            # This is the fastest method but creates a large array
            for idx in np.random.randint(0, max_idx, self.num_samples):
                yield idx

    def __len__(self):
        return self.num_samples


class SmartOptimizer():
    '''Add clipping and scheduling capabilities to a regular
       optimizer.

       Args:
           optim (Optimizer): the optimizer to use
           clipper (tuple): clipping parameters as a tuple (max_norm, norm_type). See also
               `torch.nn.utils.clip_grad_norm_` for more details
           scheduler (Scheduler): the scheduler to use
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
        dl: the dataloader (or otehr iterator) that needs to be wrapped
        skips: how many epochs should be skipped. If skips is for example 3
        the iterator is only run at every third epcoh.

    Example:
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
