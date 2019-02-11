
class BinaryAccuracy():
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

        input = (input > self.threshold).int().cpu()
        target = target.int().cpu()
        
        return (input == target).float().mean().item()
        
    