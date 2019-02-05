import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingRestartsLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule with warm restarts. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Length of the initial run (in number of epochs).
        eta_min (float): Minimum learning rate. Default: 0.
        T_mult (float): Multiplicative factor adjusting number of epochs in
        the next run that is applied after each restart. Default: 2.
        eta_mult (float): Multiplicative factor of decay in the range of
          learning rates that is applied after each restart. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, T, eta_min=0,
                 T_mult=2.0, eta_mult=1.0, last_epoch=-1):
        self.T = T
        self.eta_min = eta_min
        self.eta_mult = eta_mult

        if T_mult < 1:
            raise ValueError('T_mult should be >= 1.0.')
        self.T_mult = T_mult

        super(CosineAnnealingRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_mult == 1:
            i_restarts = self.last_epoch // self.T
            last_restart = i_restarts * self.T
            T_i = self.T

        else:
            # computation of the last restarting epoch is based on sum of geometric series:
            # last_restart = T * (1 + T_mult + T_mult ** 2 + ... + T_mult ** i_restarts)
            i_restarts = math.floor(math.log(1 - self.last_epoch * (1 - self.T_mult) / self.T,
                                             self.T_mult))
            last_restart = self.T * \
                (1 - self.T_mult ** i_restarts) / (1 - self.T_mult)
            T_i = self.T * self.T_mult ** i_restarts

        t = (self.last_epoch - last_restart) / T_i
        decay = 0.5 * (self.eta_mult ** i_restarts) * \
            (1 + math.cos(math.pi * t))

        return [decay * base_lr + (1 - decay) *
                self.eta_min for base_lr in self.base_lrs]
