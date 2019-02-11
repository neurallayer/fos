from abc import abstractmethod
from .precision import *


class Calculator():
    '''The base interface that every calculator needs to implement.
       A calculater gets metrics added and based on its purpose
       calculates a driverd value when result is invoked.

       Simple use cases inlcude calculating an average or max of the
       added values. More complex use cases include calculating F1 scores.
    '''

    @abstractmethod
    def add(self, value):
        '''Add a new value to the calculator.'''

    @abstractmethod
    def clear(self):
        '''Clear the stored values in the calculator.'''

    @abstractmethod
    def result(self):
        '''Calculate the result based on the stored values and
           return the result.
        '''

class RecentCalc(Calculator):
    '''Simple stores and returns the most recent value. So any new
       value added will overwrite the exisitng one.
    '''

    def __init__(self):
        self.value = None

    def clear(self):
        self.value = None

    def add(self, value):
        self.value = value

    def result(self):
        return self.value


class MomentumCalc(Calculator):
    '''Calculate the momentum of the value:

       value = beta*value + (1-beta)*new_value

       If it is called the first time it will use the provided
       value as initialization value::

           value = new_value

       This is a memory efficient calculator, since only one value
       needs to be stored and it still achieves silimar results as a
       moving average.

       Arguments:
          beta: What beta to use for calculating the momentum.
          disable_clear: If set to true the calculator doesn't clear between epochs.
    '''

    def __init__(self, beta=0.99, disable_clear=False):
        self.beta = beta
        self.value = None
        self.disable_clear = disable_clear

    def clear(self):
        if not self.disable_clear:
            self.value = None

    def add(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.beta * self.value + (1 - self.beta) * value

    def result(self):
        return self.value


class AvgCalc(Calculator):
    '''Calculates the average of the values.
       Memory efficient, since no history is stored.
    '''

    def __init__(self):
        self.value = 0.
        self.n = 0

    def clear(self):
        self.value = 0.
        self.n = 0

    def add(self, value):
        self.value += value
        self.n += 1

    def result(self):
        if self.n == 0:
            return None
        return self.value / self.n


class MaxCalc(Calculator):
    '''Calculates the max of the values.
       Memory efficient, since no history needs to be kept.
    '''

    def __init__(self):
        self.value = None

    def clear(self):
        self.value = None

    def add(self, value):
        if self.value is None:
            self.value = value
        elif value > self.value:
            self.value = value

    def result(self):
        return self.value


class MinCalc(Calculator):
    '''Calculates the minimum of the values.
       Memory efficient, since no history needs to be kept.
    '''

    def __init__(self):
        self.value = None

    def clear(self):
        self.value = None

    def add(self, value):
        if self.value is None:
            self.value = value
        elif value < self.value:
            self.value = value

    def result(self):
        return self.value
