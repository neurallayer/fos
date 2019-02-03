from tensorboardX import SummaryWriter

class ParamHistogram():
    '''Make histograms of the weights and gradients of 
       the parameters in the model. Every layer will get its own histograms.
       For example a Linear layer will get 4 histograms by default:
       
       - 2 Histograms for the bias parameter (values + gradient)
       - 2 Histograms for the weight parameter (values + gradient)
       
       The histograms provide insights into the model and helps to expose issues 
       like vanishing gradients. This metric uses directly TensorBoardX to 
       create the files that contain histograms. So you'll need TensorBoard to 
       view them.
       
       This metric can generate a lot of data, especially if you have many layers in your 
       model. So typically you don't run this every step (see also the skip argument below).
       
       Arguments:
           writer: which Tensorboard writer to use, if none is speficied the default SummaryWriter is be used.
           prefix: do you want to groep the metrics under a common "card" within tensorboard.
           skip: how many steps to skip until the next histograms are generated. If 
           you run this metric every step it will slown down the training.
           include_weight: Should it include the weights in the histograms
           include_gradient: Should it include the gradients int the histograms
           predictor_only: should it only assess the predictor model or the whole SupervisedModel,
           so including the loss function. Mot of the time predictor_only is all that is required 
           since loss functions don't have learnable parameters.
    '''
    
    def __init__(self, writer=None, prefix="", skip=500, include_weight=True, include_gradient=True, predictor_only=True):
        self.writer = writer
        self.prefix = prefix
        self.predictor_only = predictor_only
        self.skip  = skip
        self.include_gradient = include_gradient
        self.include_weight = include_weight
                
    def set_writer(self, subdir, writer=None):
        '''Set the summary writer to use to output the metrics'''
        self.writer = SummaryWriter("/tmp/runs/" + subdir)    

    @staticmethod
    def _get_np(param):
        return param.clone().detach().cpu().numpy()
        
    def __call__(self, model, optim):
        
        if (model.step % self.skip) != 0: return
        
        supermodel = model
        
        if self.predictor_only:
            model = model.predictor
        
        for k, v in model.named_parameters():
            
            if self.include_weight:
                name = "weight/" + k
                self._write(name, v.data, supermodel.step)
            
            if self.include_gradient and hasattr(v,"grad"):
                name = "gradient/" + k
                self._write(name, v.grad, supermodel.step)
                            
        return None

    def _write(self, name, value, step):
        # try:
        value = self._get_np(value)
        name  = self.prefix + name
        self.writer.add_histogram(name, value, step)
        #except:
        #    pass


def learning_rates(model, optim):
    '''Get the learning rates used by the optimizer'''
    
    result = []
    for p in optim.param_groups:
        result.append(p["lr"])
    return result