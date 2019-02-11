Example Notebooks
=================

One way to get famliar with Fos is to play around with some code. For that reason
the following examples in the form of Jupyter Notebooks are included:

Basic
-----
The basic notebook shows what the minimum steps that are required to train a model using
Fos. It provides a good introduction into the core concepts of Fos like Supervisor,
Trainer and Meter.

MNIST
-----
A convolutional network that is trained on the MNIST (handwritten digits) dataset. 
It also shows how to use multiple meters, visualize some metrics and save and restore 
a training session.
 

Tensorboard
-----------
This notebook demonstrates how to get some additional insight into the metrics and model. 
For example the change of gradients and weights of the model during training are visualized in 
Tensorboard. A useful technique to detect problems like vanishing gradient.

In order to see the results, you need tensorboard installed.


Inputs
------
Explains how to use Fos in combination with models that expect multiple input and/or outputs.


Google Colab
============
You can run the above notebook on your own computer once you have installed Fos. They should
run fine, even if you don't have a GPU.

But it is also possible to run them on Google Colab (Colaboratorium) without 
having to install anything. These are the links to the Google Colab notebooks:


