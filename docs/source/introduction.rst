Introduction
============
This document provides insights into the thoughts and ideas behind Fos and 
helps to better understand how to use and extend Fos. The four main guiding principles:

* **Simple** to use and extend: The basic funcitonality should be easy to grasp so you quickly can get started. 
  However when you need more advanced features it should still be possible to extend the framework to do so. Fos
  relies on good OO principles to make this possible:
  
    1. Encapsulation & Abstraction.
       Fos tries to be very concise which class performs what functionality. This makes it easier to understand 
       the framework but perhas more importantly also how to extend it. For an introduction see the 
       Components section below.

    2. Inheritance & Polymorphism. 
       Rather then providing different types of hooks to plugin functionality, you extend 
       Fos by inheriting classes like the Supervisor and override methods. Out of the box Fos already has
       many classes included that help enrich behavior like that of the optimizer.


* **Lightweight:** try to be as lightweight as possible and don't perform too much kind of magic 
  under the hood. It is important that it is easy to grasp what a framework is actualy doing so it 
  doesn't work against you at times. Fos uses plain PyTorch objects where possible and only adds features
  where it really adds value.


* **Insightful:** perhaps the most challanging task of developing a neural network is investigating why it 
  doesn't perform as expected. A framework like Fos should facilitate getting the required insights of 
  what is going on and what can be done to improve the peformance. 
  
  Below is an example of the type of insights Fos can generate (in this case the histograms of the gradients 
  of the model plotted over time).
  
  .. image:: /_static/img/tensorboard_gradients.png
 


* **Repeatable:** once you are developing models, it is important that you can easily repeat results and
  can continue where you left off. So Fos for example makes it easy to store and save the trianing state.


Quick Start
===========
The following code shows the steps required to train a model. In this example we'll use the resnet18
model that comes with PyTorch vision.

First import the required modules, including the three classes from Fos we are going to use::

    import torch
    import torch.nn.functional as F
    from torchvision.models import resnet18 
    from fos import Supervisor, Trainer, NotebookMeter

Then we create the model we want to train, an optimizer and the loss function::

   predictor = resnet18()
   optim     = torch.optim.Adam(predictor.parameters())
   loss      = F.binary_cross_entropy_with_logits

Finally we create the supervised model, a meter and the trainer::

   model   = Supervisor(predictor, loss)
   meter   = NotebookMeter()
   trainer = Trainer(model, optim, meter)

And we are ready to start the training. We create some dummy data in this example that emulates 
the input images and target values en then run the trainer::

   dummy_data = [(torch.randn(4,3,224,224), torch.rand(4,1000).round()) for i in range(10)]
   trainer.run(dummy_data, epochs=5)

As you can see, only a few lines of code are required to get started. And the above if a fully
working example, no steps skipped. The following section gives an highlevel overview of the Fos 
components and their purpose. If you want to get more details, best to dive into the 
package reference documentation.


Components
==========

Supervisor
----------
Supervisor is a model that adds a loss function to the model that you want to train.
So you create an instance by providing both the model you want to train (we refer to this as the predictor) 
and the loss function::

    model = Supervisor(predictor, loss_fn)

Under the hood, the forward of the supermodel would look something like this::

    def forward(self, x, target):
        y_pred = predictor(x)
        loss   = loss_fn(y_pred, target)
        return loss

The Supervisor has also additional methods to train and validate a batch and is used by the Trainer to train the model.
It is the Supervisor responsibility to perform both the forward and backward step. And the Supervisor optionally invokes the additional metrics to get more insights how the model is performing. However is is the trainer that updates 
the model by invoking `optimizer.step`.

The provided Supervisor implementation can handle most scenarios, but you can alway extend it to 
cather for specific use cases.

Metric
------
A metric is nothing more then a plain Python function that can be added to a Supervisor or a Trainer to get extra insights into
the performance of your model. There are two types of metrics support:

1) Metrics that evaluate the prediction vs target values. These can be passed as an argument when you create a Supervisor. 
2) metrics that evaluate the model itself. These can be passed as an argument when you create a Trainer.

Metrics are optional and if you don't provide any, only the loss value will be added as a metric.

Meter
-----
A meter captures the generated metrics and displays them by for example printing results in a Jupyter Notebook or 
logging them to a file. Whenever the trainer is done with a training step, it will retrieve the generated metrics and hand them
over to the meter (meter.update).


Read more about meters (and calcuators) at meters.rst

Trainer
-------
The trainer is the coponent that glues all other components together and responsible for running the training epochs. 
The trainer contains the loops that go over the provided data (trainer.run). 

To initiate a trainer you need to provide at least a supermodel, optimizer and meter::

    trainer = Trainer(model, optimizer, meter)
    
And then to train for a number of epochs you need to provide the data::

    trainer.run(data, validation_data, epochs=10)

The diagram below shows how the components are linked to each other.

.. image:: /_static/img/logical_components.png


Flow
====
The following diagram shows the interactin between the various components when you invoke trainer.run:

.. image:: /_static/img/logical_flow.png


Inspiration
===========
There are many other frameworks available, some of which also support PyTorch. Many of them
have been source of inspiration for Fos, but there are also some differences:

- `PyTorch Ignite`: very flexible and extensible framework while staying lightweight. Ignite has a more 
  functional API and relies to registring handlers to extend functionality where Fos uses OO principles.  
  
- `FastAI`: Includes many best practices out of the box behind the API and of course there are also 
  excellent courses to accompyning it. Fos does by default less magic behind the scene and the way to 
  include these best practices in your training is to use one of more the specialized classes.

- `Keras`: Unfortunatly no support for PyTorch, but nice API and very easy to use. One of key differences 
  is that Keras abstracts most of the underlying machine learning engine (by design), where as 
  Fos augments the engine (PyTorch) rather than hiding it.
  
- `Chainer`: Excellent API that also uses a OO approach. It has however its own ML engine and not 
  PyTorch (although PyTorch and other engines borrowed a lot of their API's from Chainer)


As always, give them a spin and see which framework suits your way of working best. 


Contribution
============
If you want to help out, we appreciate all contributions. 
Please see the `Contributing Guidelines <https:github.com/innerlogic/fos/CONTRIBUTING.rst>`__ for more information.

And ofcourse, PRs are welcome :)= 


