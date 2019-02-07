
class ConfusionMetric:
    '''Calculate the TP, FP, TN and FN for the predicted classes. 
       There are several calculators available that use these base metrics 
       to calculate for example recall, precision or beta scores. 
    
       Args:
           threshold: what threshold to use to say a probabity represents a true label 
           sigmoid: should a sigmoid be applied before determining true labels
           
        Example:
            metric = ConfusionMetric(threshold=0.5, sigmoid=True)
            model  = SuperModel(..., metrics = {"tp": metric})
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

        tp = (y * t).float().sum(dim=0)
        tn = ((1 - y) * (1 - t)).float().sum(dim=0)
        fp = (y > t).float().sum(dim=0)
        fn = (y < t).float().sum(dim=0)

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }
