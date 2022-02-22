Installation
============

Required Dependencies
---------------------
- Python 3.8 or later
- OpenSeesPy
- opsvis
- numpy
- matplotlib

Instructions
------------
The easiest way to install `ospgrid` is to use the python package index: ::

    pip install ospgrid

For users wishing to develop: ::

    git clone https://github.com/ccaprani/ospgrid.git
    cd ospgrid
    pip install -e .
    
For contributions, first fork the repo and clone from your fork. `Here <https://www.dataschool.io/how-to-contribute-on-github/>`_ is a good guide on this workflow.

Anaconda
--------
It's best to install OpenSeesPy into its own environment, due to its dependencies, and potential clashes with other packages.
So in turn, execute: ::

    conda create --name ospgrid python=3.8  

And then install as above.

And as a reminder, if you are using `spyder` as your IDE, don't forget to also install the spyder kernels in the new environment: ::

    conda install spyder-kernels=2.1


Tests
-----
`ospgrid` comes with ``pytest`` functions to verify the correct functioning of the package. 
Users can test this using: ::

    python -m pytest

from the root directory of the package.
