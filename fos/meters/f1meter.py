import torch


class FScoreCalc:
    '''F macro score is both a metric and a calculator.
       The default is F1 (beta=1), but any possitve real value for beta is possible.

       Example:
           f1 = FScoreCalc()
           model = SupervisedModel(.... metrics={"f1": f1})
           meter = PrintMeter({"f1":f1})

    '''

    def __init__(self, threshold=0.5, beta=1, sigmoid=False, eps=1e-8):
        self.reset()
        self.sigmoid = sigmoid
        self.beta = beta
        self.threshold = threshold
        self.history = []
        self.eps = eps

    def __call__(self, y, t):

        assert y.shape == t.shape, "Shapes of target and predicted values should be the same for F1 Meter"

        y = y.flatten(1)
        t = t.flatten(1)

        if self.sigmoid:
            y = y.sigmoid()

        if self.TP is None:
            self.init(t.shape[-1])

        y = (y > self.threshold).int().cpu()
        t = t.int().cpu()

        self.TP += (y * t).float().sum(dim=0)
        self.FP += (y > t).float().sum(dim=0)
        self.FN += (y < t).float().sum(dim=0)
        return True

    def add(self, value):
        pass

    def init(self, n):
        self.TP = torch.zeros(n)
        self.FP = torch.zeros(n)
        self.FN = torch.zeros(n)

    def result(self):
        if self.TP is None:
            return 0

        beta2 = self.beta**2
        score = (1 + beta2) * self.TP / ((1 + beta2) * self.TP +
                                         (beta2) * self.FP + self.FN + self.eps).mean()
        return score.item()

    def clear(self):
        self.TP = None
        self.FP = None
        self.FN = None
