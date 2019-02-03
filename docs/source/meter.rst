Meter & calculator
==================
When you train a model, you'll need of course to see what is gong on. This is where the Meter kicks in. 
Its main responsibility is to capture metrics and somehow display them. This can be printing them in 
a ``Jupyter Notebook``, but also logging them to a ``tensorboard`` compatible file.

There are sever al Meters provided out od the box that will get you started. 

The Meter has to implement the following methods::

    def update(value):
        ...
        
    def display(ctx):
        ...
    
    def reset():
         ...

Many of the out of the box meters delegate the actual calculations to a calculator. So for example the Notebook meter
delegates the calculations of the average values to the AgvCalc.  

