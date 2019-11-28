Introduction
============
**FOS** is a Python framework that makes it easy to develop neural network models 
in PyTorch. Some of its main features are:

* Less boilerplate code required, see also the example below.
* Lightweight and no magic under the hood that might get in the way.
* You can extend Fos using common OO patterns.
* Get the insights you need into the performance of the model.


Installation
============
You can install FOS using pip::

    pip install fos
    
Or alternatively from the source::

    python setup.py install
    
Fos requires Python 3.5 or higher.


Usage
=====
Training a model, requires just a few lines of code. First create the model, 
optimizer and loss function that you want to use, creating plain PyTorch objects::

   net   = resnet18()
   optim = Adam(predictor.parameters())
   loss  = F.binary_cross_entropy_with_logits

Then create the FOS objects that will take care of the training and output::

   workout   = Workout(predictor, loss, optim)
  

And we are ready to start the training::

   trainer.fit(train_data, valid_data, epochs=5)


Examples
========
You can find several example Jupyter notebooks `here <https://github.com/innerlogic/fos/examples>`_, 
or even more convenient try them directly in a Google Colab environment:

    1) Basic Example
    2) MNIST example


Contribution
============
If you want to help out, we appreciate all contributions. 
Please see the [contribution guidelines]() for more information.

As always, PRs are welcome :)= 