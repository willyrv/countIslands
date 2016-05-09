# estimate_islands
Estimating the number of islands under an n-island model.

This is a work in progress. The aim of the scripts in this repository is to estimate the number of islands from the number of differences. The underlying model is the n-island model of Wright. Three parameters need to be estimated from data: the number of islands (or demes, denoted n), the migration rate (denoted M) and the scaled mutation rate (denoted theta).

Prerequisites:
--------------
- Python 2.7
- scipy, numpy, matploblib
- cython

Compiling
---------
python ./setup.py build_ext --inplace

Trace the plot of the likelihood
--------------------------------
It is possible to do a 3d-plot of the likelihood function for differences values of *(n, M)* and a fixed value of *theta*. First open the file *trace_likelihood_3d.py* and set the corresponding parameters:
- the input file
- range of values for *n*
- range of values for *M*
- the value of *theta*

Then do:
python ./trace_likelihood_3d.py
