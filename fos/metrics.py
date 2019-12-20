'''
This module contains the various metrics that can used to monitor the progress during training.
Any loss function can be used as a metric, however not every metric is a valid loss function.
'''
import math
from abc import abstractmethod
from torch import Tensor
# pylint: disable=C0103, W0612, R0903

class Metric():
    '''This is the interface that needs to be implemented
       by a metric class.
    '''

    @abstractmethod
    def __call__(self, y_pred, y_true):
        '''Calculate the metric value. Typically the y_pred and y_true are Tensors, but in more
           complex scenarios they also can be tuples of Tensors (for example when a model
           returns multiple tensors).

           Args:
               y_pred: the predicted output
               y_true: the true output (labels)
        '''


class BinaryAccuracy(Metric):
    '''Calculate the binary accuracy score between the predicted and target values.

    Args:
        threshold (float): The threshold to use to determine if the input value is 0 or 1
        sigmoid (bool): should sigmoid activatin be applied to the input
    '''

    def __init__(self, threshold=0., sigmoid=False):
        self.sigmoid = sigmoid
        self.threshold = threshold

    def __call__(self, pred: Tensor, target: Tensor):
        pred = pred.flatten(1)
        target = target.flatten(1)

        assert pred.shape == target.shape, "Shapes of target and predicted values should be same"

        if self.sigmoid:
            pred = pred.sigmoid()

        pred = (pred > self.threshold).int()
        target = target.int()

        return (pred == target).float().mean()


class SingleClassAccuracy(Metric):
    '''Accuracy for single class predictions like MNIST.
       Label is expected in the shape [Batch] and predictions [N x Batch]
    '''
    def __init__(self, activation=None):
        self.activation = activation

    def __call__(self, yp: Tensor, y: Tensor):
        if self.activation is not None:
            yp = self.activation(yp)

        (_, arg_maxs) = yp.max(dim=1)
        result = (y == arg_maxs).float().mean()
        return result

def _get_metrics(workout, metric):
    keys = list(workout.history[metric].keys())
    keys.sort()
    return keys, [workout.history[metric][k] for k in keys]

def plot_metrics(plt, workout, metrics):
    '''Plot metrics collected during training'''
    for metric in metrics:
        x_data, y_data = _get_metrics(workout, metric)
        plt.plot(x_data, y_data)

    plt.xlabel("steps")
    plt.ylabel("values")
    plt.legend(metrics)
    return plt.show()

def _get_sum(tensor: Tensor) -> float:
    return tensor.float().sum(dim=0).mean()


class ConfusionMetric(Metric):
    '''Calculate the TP, FP, TN and FN for the predicted classes.
       There are several calculators available that use these base metrics
       to calculate for example recall, precision or beta scores.

       Args:
           threshold: what threshold to use to say a probabity represents a true label
           sigmoid: should a sigmoid be applied before determining true labels

       Example usage:

       .. code-block:: python

            metric = ConfusionMetric(threshold=0.5, sigmoid=True)
            model  = Supervisor(..., metrics = {"tp": metric})
            meter  = TensorBoardMeter(metrics={"tp": RecallCalculator()})
    '''

    def __init__(self, threshold=0., sigmoid=False):
        self.sigmoid = sigmoid
        self.threshold = threshold

    def __call__(self, y: Tensor, t: Tensor):

        y = y.flatten(1)
        t = t.flatten(1)

        assert y.shape == t.shape, "Shapes of target and predicted values should be same"

        if self.sigmoid:
            y = y.sigmoid()

        y = (y > self.threshold).int().cpu()
        t = t.int().cpu()

        return (
            _get_sum(y * t), #tp
            _get_sum(y > t), #fp
            _get_sum(y < t), #fn
            _get_sum((1 - y) * (1 - t)) #tn
        )



class Precision(ConfusionMetric):
    '''Precision metric'''

    def __call__(self, y, t):
        tp, fp, fn, tn = super().__call__(y, t)
        return tp/(tp+fp)


class Recall(ConfusionMetric):
    '''Recall metric'''

    def __call__(self, y, t):
        tp, fp, fn, tn = super().__call__(y, t)
        return tp/(tp+fn)


class F1Score(ConfusionMetric):
    '''F1 Score metric'''

    def __call__(self, y, t):
        tp, fp, fn, tn = super().__call__(y, t)
        return 2*tp/(2*tp+fn+fp)


class MCC(ConfusionMetric):
    '''MCC metric'''

    def __call__(self, y, t):
        tp, fp, fn, tn = super().__call__(y, t)
        return (tp*tn - fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp*(tn+fn)))
