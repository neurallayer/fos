from abc import abstractmethod


class Metric():
    '''This is the interface that needs to be implemented
       by a metric class.
    '''

    @abstractmethod
    def __call__(self, y_pred, y_true):
        '''calculate the metric

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

    def __call__(self, input, target):
        input = input.flatten(1)
        target = target.flatten(1)

        assert input.shape == target.shape, "Shapes of target and predicted values should be same"

        if self.sigmoid:
            input = input.sigmoid()

        input = (input > self.threshold).int()
        target = target.int()

        return (input == target).float().mean()


def _get_metrics(workout, metric):
    keys = list(workout.history[metric].keys())
    keys.sort()
    return keys, [workout.history[metric][k] for k in keys]


def plot_metrics(plt, workout, metrics):
    for metric in metrics:
        X, Y = _get_metrics(workout, metric)
        plt.plot(X, Y)

    plt.xlabel("steps")
    plt.ylabel("values")
    plt.legend(metrics)
    return plt.show()
