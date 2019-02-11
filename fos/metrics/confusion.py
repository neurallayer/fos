

def _get_sum(tensor):
    return tensor.float().sum(dim=0)

class ConfusionMetric:
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
        
    def __call__(self, y, t):

        y = y.flatten(1)
        t = t.flatten(1)

        assert y.shape == t.shape, "Shapes of target and predicted values should be same"

        if self.sigmoid:
            y = y.sigmoid()

        y = (y > self.threshold).int().cpu()
        t = t.int().cpu()

        return {
            "tp": _get_sum(y * t),
            "fp": _get_sum(y > t),
            "fn": _get_sum(y < t),
            "tn": _get_sum((1 - y) * (1 - t))
        }
