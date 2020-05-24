Callbacks
=========
The meters main function is to capture the generated metrics and display them. Display can be printing them in 
a ``Jupyter notebook`` cell, but also logging them to a ``tensorboard`` file or inserting them in a database.

There are several Meters provided out of the box that will get suit most needs. Many of the out of the box meters delegate the actual calculations to a calculator. So for example the Notebook meter delegates by default the calculations of the values to the AgvCalc. 


Example::

    # capture all metrics and use the AvgCalc
    meter = Notebookmeter()
    
    # capture only "acc" metric and use the MomentumCalc
    meter = MemoryMeter(metrics={"acc": MomentumCalc()})
    
    # capture all metrics except "val_loss" and "loss" metrics
    meter = PrintMeter(exclude=["val_loss", "loss"])
 

.. currentmodule:: fos.callbacks

.. autoclass:: Meter
    :members:
    
.. autoclass:: MultiMeter
    
.. autoclass:: BaseMeter
.. autoclass:: NotebookMeter
.. autoclass:: PrintMeter
.. autoclass:: MemoryMeter
.. autoclass:: TensorBoardMeter

