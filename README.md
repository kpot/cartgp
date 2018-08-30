CartGP - Cartesian Genetic Programming Library for C++ and Python
-----------------------------------------------------------------

CartGP is a very simple and minimalistic C++/Python library implementing 
[Cartesian Genetic Programming][1] (CGP).
The library currently supports classic form of CGP where nodes are arranged
into a grid and no recurrent connections are allowed.

Python binding
==============
Python version of the library can be installed using standard
`python setup.py install` or `python pip .` commands, from the project's
directory. You will also need to have CMake and a C++11-compatible compiler
installed.

Check [this jupyter notebook](./examples/python/symbolic_regression.ipynb)
to see how to use the library from Python.


C++ Interface
=============
CartGP is a header-only library that does not require building. Simply add
files from [cpp/include](./cpp/include) to your project and
`#include <cartgp/genotype.h>` to your code. C++11 or newer is required.


Potential issues
================
If you see error `Symbol not found: __Py_ZeroStruct` on `import pycartgp`,
check that you're running your program using the same Python interpreter
you used for installing PyCartGP.

[1]: http://www.cartesiangp.co.uk/ "Cartesian Genetic Programming"
