import torch 

class BaseMacroCalculator():
    '''Base for other macro calculations. Should be used together with TPMetric
        
        Example:
        
        model = SupervisedModel(... metric={"tp":TPMetrics})
        meter = PrintMeter({"tp":PrecisionCalculator()})
        trainer(.... meters=[meter])
    '''
    
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.clear()
    
    def clear(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.n  = 0

    def add(self, value):
        self.tp += value["tp"]
        self.fp += value["fp"]
        self.tn += value["tn"]
        self.fn += value["fn"]
        self.n  += 1
            
    def result(self):
        ...



class PrecisionCalculator(BaseMacroCalculator):
    '''Calculates the precision: 
            precision = tp/(tp+fp)
    '''

    def result(self):
        if self.n == 0: return None
        result = self.tp/(self.tp+self.fp+self.eps)
        return result.mean()


class RecallCalculator(BaseMacroCalculator):
    '''Calculates the recall: 
            precision = tp/(tp+fn)
    '''
    
    def result(self):
        if self.n == 0: return None
        result = self.tp/(self.tp+self.fn+self.eps)
        return result.mean()


class BetaCalculator(BaseMacroCalculator):
    '''Calculates F Beta score, default is F1 (beta=1)
    '''

    def __init__(self, beta=1, eps=1e-8):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def result(self):
        if self.n == 0: return None
        beta2 = self.beta**2
        score = ((1+beta2)*self.tp/((1+beta2)*self.tp + (beta2)*self.fp + self.fn + self.eps)).mean()
        return score.item()
        