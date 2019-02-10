
class BinaryAccuracy():
    '''Calculate the binary accuracy score between the predicted and target values'''

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
        
        return (y == t).float().mean().item()
        
    