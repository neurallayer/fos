Meters
======
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
 

.. currentmodule:: fos.meters

.. autoclass:: Meter
    :members:
    
.. autoclass:: MultiMeter
    
.. autoclass:: BaseMeter
.. autoclass:: NotebookMeter
.. autoclass:: PrintMeter
.. autoclass:: MemoryMeter
.. autoclass:: TensorBoardMeter


Calculators
===========
Calculators are used by several meters to perform the required calculations. Typically during training and validation epoch,
the calculator get updated with the values of a metric. When the display method is invoked, the calulator calculates the resulting value, like the average of that metric so far and return this to the meter.

.. currentmodule:: fos.calc

.. autoclass:: Calculator
    :members:

Standard Calculators
--------------------
A number of calulators are included that perform basic calculations like the average or momentum over
a series of metric values.

.. autoclass:: AvgCalc
.. autoclass:: MinCalc
.. autoclass:: MaxCalc
.. autoclass:: RecentCalc
.. autoclass:: MomentumCalc

Precision Calculators
---------------------
There are also calculators included that perform precision and recall type of calculations:

.. autoclass:: PrecisionCalculator
.. autoclass:: RecallCalculator
.. autoclass:: BetaCalculator
