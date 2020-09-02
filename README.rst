.. image:: https://travis-ci.com/neurallayer/fos.svg?branch=master
    :target: https://travis-ci.com/neurallayer/fos


Introduction
============
**FOS** is a Python framework that makes it easy to develop neural network models 
in PyTorch. Some of its main features include:

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
    
Fos requires Python 3.6 or higher.


Usage
=====
Training a model, requires just a few lines of code. First create the model, 
optimizer and loss function that you want to use, using normal PyTorch code::

   model = resnet18()
   optim = Adam(model.parameters())
   loss = F.binary_cross_entropy_with_logits

Then create the FOS workout that will take care of the training and output::

   workout = Workout(net, loss, optim)
  
And we are ready to start the training::

   workout.fit(train_data, valid_data, epochs=5)


Examples
========
You can find several example Jupyter notebooks `here <https://github.com/neurallayer/fos/tree/master/examples>`_ 

You can also run them on Google Colab directly:

* `Basic https://colab.research.google.com/github/neurallayer/fos/blob/master/examples/basic_fos.ipynb`
* `MINST https://colab.research.google.com/github/neurallayer/fos/blob/master/examples/mnist_fos.ipynb`
* `Inputs https://colab.research.google.com/github/neurallayer/fos/blob/master/examples/inputs_fos.ipynb`
* `Tensorboard https://colab.research.google.com/github/neurallayer/fos/blob/master/examples/tensorboard_fos.ipynb`



Contribution
============
If you want to help out, we appreciate all contributions. 
Please see the contribution guidelines for more information.

As always, PRs are welcome :)= 
