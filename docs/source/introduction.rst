Introduction
============
This document provides insights into the thoughts and ideas behind FOS and 
helps to better understand how to use and extend FOS. The main guiding principles are

*  **Simple** to use and extend. The basic funcitonality should be easy to grasp so you quickly can get started. 
   However when you need more advanced features it should still be possible to extend the framework to do so. FOS
   relies on good OO principles to make this possible:
  
    1. Encapsulation & Abstraction.
       FOS tries to be very concise which class performs what functionality. This makes it easier to understand 
       the framework but perhas more importantly also how to extend it. For an introduction see the 
       Components section below.
       
    2. Inheritance & Polymorphism. 
       Rather then providing different types of hooks to plugin functionality, you extend 
       FOS by inheriting classes like the Supervisor and override methods. Out of the box FOS already has
       many classes included that help enrich behavior like that of the optimizer.
       
*  **Lightweight**: try to be as lightweight as possible and don't perform too much kind of magic 
   under the hood. It is important that it is easy to grasp what a framework is actualy doing so it 
   doesn't work against you at times. FOS uses plain PyTorch objects where possible and only adds features
   where it really adds value.
                
*  **Insightful**: perhaps the most challanging task of developing a neural network is investigating why it 
   doesn't perform as expected. FOS facilitates getting the required insights of what is going on and 
   what can be done to improve the peformance. 
  
   Below is an example of the type of insights FOS can generate (in this case the histograms of the gradients 
   of the model plotted over time).
  
   .. image:: /_static/img/tensorboard_gradients.png
   
*  **Repeatable**: once you are developing models, it is important that you can easily repeat results and
   can continue where you left off. So FOS for example makes it easy to store and save the trianing state.


Quick Start
===========
    The following code shows the steps required to train a model. In this example we'll use the resnet18
    model that comes with PyTorch vision.

    First import the required modules, including the three classes from FOS we are going to use::

        import torch
        import torch.nn.functional as F
        from torchvision.models import resnet18 

        from fos import Workout
        from fos.meters import NotebookMeter

    Then we create the model we want to train, and the loss function::

       net   = resnet18()
       loss  = F.binary_cross_entropy_with_logits

    Finally we create the supervised model, a meter and the trainer::

       workout   = Workout(net, loss)
       meter     = NotebookMeter()
       

    And we are ready to start the training. We create some dummy data in this example that emulates 
    the input images and target values. Typically this would be a dataloader in a real scenario. 
    And then we can run the trainer::

       dummy_data = [(torch.randn(4,3,224,224), torch.rand(4,1000).round()) for i in range(10)]
       workout.fit(dummy_data, epochs=5, callbacks = [meter])

    So only a few lines of code are required to get started. And the above if a fully
    working example, no steps skipped. The following section gives an highlevel overview of the FOS 
    components we just used and their purpose. If you want to get more details, best to dive into the 
    package reference documentation or one of the notebooks.


Components
==========

Workout
-------
Workout is the central component that adds a loss function and optimizer to the model that you want to train.
So you create an instance by providing the model you want to train, the loss function, the optimizer and metrics::

    workout = Workout(model, loss_fn, optim, acc=metric1, someother=metric2)


Training
--------
After you created the workout, the training can start::

    trainer.run(data, validation_data, epochs=10, callbacks=[NotebookMeter()])


Metric
------
A metric is nothing more then a plain Python function that can be added to a Workout to get extra insights into
the performance of your model. Metrics are optional and if you don't provide any, only the loss value will be added as a metric.


Callback
--------
A callback captures the generated metrics and displays them by for example printing results in a Jupyter Notebook or 
logging them to a file. Whenever the training is done with a training step, it will retrieve the generated metrics and hand them
over to the meter. Read more about meters at callbacks.rst


Inspiration
===========
There are many other frameworks available, some of which also support PyTorch. Many of them
have been source of inspiration for FOS, but there are also some differences:

- ``PyTorch Ignite``: very flexible and extensible framework while staying lightweight. Ignite has a more 
  functional API and relies to registring handlers to extend functionality where FOS uses OO principles.  
  
- ``FastAI``: includes many best practices out of the box behind the API and of course there are also 
  excellent courses to accompyning it. FOS does by default less magic behind the scene and the way to 
  include these best practices in your training is to use one of more the specialized classes.

- ``Keras``: unfortunatly no support for PyTorch, but nice API and very easy to use. One of key differences 
  is that Keras abstracts most of the underlying machine learning engine (by design), where as 
  FOS augments the engine (PyTorch) rather than hiding it.
  
- ``Chainer``: excellent API that also uses a OO approach. It has however its own ML engine and not 
  PyTorch (although PyTorch and other engines borrowed a lot of their API's from Chainer)


As always, give them a spin and see which framework suits your way of working best. 


Contribution
============
If you want to help out, we appreciate all contributions. 
Please see the `Contributing Guidelines <https:github.com/innerlogic/fos/CONTRIBUTING.rst>`__ for more information.

And ofcourse, PRs are welcome :)= 


