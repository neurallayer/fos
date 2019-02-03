Usage
=====

This directory contains the unittests for *Fos*.
Fos uses `PyTest` for unit testing. The below 
code snippets assume you are in the Fos sourcecode 
root directory. 

If you have already installed Fos, 
the unittests can be easily run by starting PyTest from 
the test directory::

   cd test
   pytest


PyTest will automtically detect the unittest files and execute them all.If you didn't 
install Fos yet and only have the source code, you'll nee to add 
the Fos root directory to the PYTHONPATH variable::

   export PYTHONPATH=$PYTHONPATH:`pwd`
   cd test
   pytest

Structure
=========

Every main class/component has its own unittest file. Smaller classes are grouped
into one unittest file (often my module name)