Metrics
=======
Metrics are a great way to get extra information about the performance of the model besides the loss. Metrics are plain 
Python functions that can be passed as an argument when you create the SuperModel. You pass them as a dictionary o which 
the key is name under which the metric wil be known and the value is the metric function iself. For example::

    SuperModel(..., metrics={"acc": accuracy})


There are two different types of metrics that are being supported by Fos:

1. Prediction Metrics. These metrics are invoked after a prediction is made (train and validation)::

        def metric(y: Tensor, t:Tensor):
            ...
    
   The prediction metrics will be calculated both during the training and validation iterations. The values are stored in the output attribute of the SuperModel, with the difference that the metric of the validation iteration is prepended with "val\_". 
    
2. Model Metrics. Model metrics provide insights into the model itself and are only called during training::

        def metric(model: SuperModel, optim:Optimizer):
            ...
            
   Model metrics are invoked just after the backward pass but before the optimizer step. So the model has the gradients calculated but not yet applied. Model metrics come in very handy when investigating why the model is not performing as expected. Some common issues that can be detected:
    
        - vanishing and exploding gradients
        - incorrectly initialized weights
        

Examples
========

The followig snippet shows how to implement a simple accuracy metric (predection metric)::

    def accuracy(y, t):
        ...
        
    model = Supermodel(predictor, optim, metrics={"acc": accuracy})
    

And an example of a model metric::

    def fc_metric(model):
        # Returns the avg weight of the fully connected layer 
        return model.predictor.fc.weight.avg().item()
        
    model = Supermodel(predictor, optim, model_metrics={"fcweight": fc_metric})


Classes
=======
