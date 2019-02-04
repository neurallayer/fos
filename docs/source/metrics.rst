Metrics
=======
Metrics are the way to get extra information about the performance of the model besides the loss. Metrics are plain 
Python functions that can be passed as an argument when you create the SuperModel or Trainer. You pass them as a dictionary 
of which the key is name under which the metric wil be known and the value is the metric function iself. For example::

    model   = SuperModel(..., metrics={"acc": accuracy})
    trainer = Trainer(..., metrics={"avgweights": my_weights_metric})


There are two different types of metrics that are being supported by Fos:

1. Prediction Metrics. These metrics are invoked after a prediction is made::

        def metric(y: Tensor, t:Tensor):
            ...
    
   The prediction metrics will be calculated both during the training and validation iterations. Prediction
   metrics need to be passed as an argument to the SuperModel.
    
2. Model Metrics. Model metrics provide insights into the model itself and are only called during training::

        def metric(model: SuperModel, optim:Optimizer):
            ...
            
   Model metrics are invoked just after the backward pass but before the optimizer step. So the model has the gradients calculated but not yet applied. Model metrics come in very handy when investigating why the model is not performing as expected. Some common issues that can be detected:
    
        - vanishing and exploding gradients
        - incorrectly initialized weights
        
   Model metrics need to be passed as an argument to the Trainer.


Examples
========

The followig snippet shows how to implement an accuracy metric (of the type predection metric)::

    def accuracy(y, t):
        y = torch.argmax(y, dim=-1)
        return (y == t).float().mean().item()
        
    model = Supermodel(predictor, optim, metrics={"acc": accuracy})
    

And an example of a weight metric (of the type model metric)::

    def fc_metric(model, optim):
        # Returns the avg weight of the fully connected layer 
        return model.predictor.fc.weight.avg().item()
        
    trainer = Trainer(model, optim, model_metrics={"fcweight": fc_metric})


Classes
=======
