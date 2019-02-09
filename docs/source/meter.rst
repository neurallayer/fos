Meter & calculator
==================
The meters main responsibility is to capture metrics and display them. Display can be printing them in 
a ``Jupyter notebook``, but also logging them to a ``tensorboard`` file or storing them in a database.

The Meter has to implement the following methods::

    def update(value):
        ...
        
    def display(ctx):
        ...
    
    def reset():
         ...


There are several Meters provided out of the box that will get you started. Many of the out of the box meters delegate the actual calculations to a calculator. So for example the Notebook meter delegates by default the calculations of the values to the AgvCalc. 


Examples
--------
    # capture all metrics and use the AvgCalc
    meter = Notebookmeter()
    
    # capture only `acc` metric and use the MomentumCalc
    meter = MemoryMeter(metrics={"acc": MomentumCalc()})
    
    # capture all metrics except `val_loss` and `loss` metrics
    meter = PrintMeter(exclude=["val_loss", "loss"])