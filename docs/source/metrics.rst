.. currentmodule:: fos.metrics

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


Prediction Metrics
------------------
Prediction metrics provide insights how the predictions are performing compared to the target values.
A prediction metric is a plain Python function with the following signature::

      def metric(input: Tensor, target:Tensor):
            ...

So most lost functions that come with PyTorch can also double as a prediciton metric. The following snippet shows 
how to implement a simple custom accuracy metric::

    def accuracy(y, t):
        y = torch.argmax(y, dim=-1)
        return (y == t).float().mean().item()
        
        
To use a metric, it needs to be passed as an dictionary argument when creating a Supervisor::  
        
    model = Supervisor(predictor, optim, metrics={"acc": accuracy})
    
The key of the dictionary ("acc") will be the name of the metric. When the metric is invoked during the validaiton phase it 
will be prepended with `val_`. So after one step, the followig two metrics will be published: `acc` and 'val_acc'. 

.. autoclass:: BinaryAccuracy

.. autoclass:: ConfusionMetric


Model Metrics
-------------
Model metrics dont evaluate prediction vs target but instead look at the model itself to determine 
how the training is doing. Typically a model metric would look at weights and gradients of the 
parameters in the model and provide some statistics that help to identify possible performance problems
like vanishing gradients.

An example of a model metric::

    def fc_metric(model, optim):
        # Returns the avg weight of the fully connected layer 
        return model.predictor.fc.weight.avg().item()
        
    trainer = Trainer(model, optim, model_metrics={"fcweight": fc_metric})

.. autoclass:: ParamHistogram

