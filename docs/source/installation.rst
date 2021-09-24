.. _installation:

=======================
How to Install CARL
=======================

.. role:: bash(code)
    :language: bash

First clone our GitHub repository:

.. code-block:: bash
    git clone https://github.com/automl/CARL.git
    cd clone

We recommend installing within a virtual environment:

.. code-block:: bash

    conda create -n carl python=3.9
    conda activate carl

Then you may install the base version:
.. code-block:: bash

    cd carl
    pip install

Alternatively, you can also include more environments in your installation:
.. code-block:: bash

    pip install -e [box2d, brax, rna, mario]

Or the dependencies for our experiments:
.. code-block:: bash

    pip install -e [experiments]

You should now have CARL installed.
The code itself can be found in the 'src' folder.
