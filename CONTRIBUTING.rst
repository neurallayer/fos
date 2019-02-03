Contributing to Fos
-------------------

If you are interested in helping out with Fos, your contributions are more than welcome.
Typically they will fall into one of the following two categories:

1. You want to implement a new feature.
    - Post about your intended feature and how it would help you and others. 
      Once we agree that the plan looks good, go ahead and implement it.
    
2. You found a bug and want to fix it.
    - Look at the outstanding issues here: `<https://github.com/innerlogic/fos/issues>`_
    - If the bug is not mentioned yet, please raise a new issue.
    - Once confirmed, you can implement the bug-fix.
    
Once you finish implementing a feature or bugfix, please send a Pull Request to
`<https://github.com/innerlogic/fos>`_


.. seealso:: If you are not familiar with creating a Pull Request, here are some guides:
     - `<http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request>`_
     - `<https://help.github.com/articles/creating-a-pull-request/>`_


Unit testing
------------
Fos testing is located under `test/`. Run the entire test suite with PyTest::

    cd test
    pytest

Writing documentation
---------------------
Fos uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for formatting docstrings. Length of line inside docstrings block must be limited to 80 characters 
to fit into Jupyter documentation popups.