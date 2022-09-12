==========================================
BetabinGLM: Beta-binomial regression model
==========================================

Installation
============

You can install from this GitHub repository for the latest (often development) 
version by following the command line.

.. code-block:: bash

  pip install -U git+https://github.com/StatBiomed/BetabinGLM

In either case, add ``--user`` if you don't have write permission for your 
Python environment.


Quick start
===========

.. code-block:: python

  from BetabinGLM import betabin

  model = betabin(x, y, fit_intercept = True, method = "SLSQP")

Where x is an array of n * w dimensions, y is an array of n * 2 dimensions. n is the sample size, w is the number of independent variables. The columns 1 and 2 of y represent the numbers of positive and negative observation respectively.

Constants for intercept are added by default. For optimization, this package supports all methods in sp.optimize.minimize. SLSQP is used by default.

To obtain the Log-Likelihood:

.. code-block:: python

  model.LL
  
  
Try testing a few samples, if it doesn't provide a satisfying result: 1) errors occured (e.g. the initialization using binomial regression encounters a nan value problem) 2) Provides unaccurate result (e.g. returns a groups of 0 in parameters or provides a very small log-likelihood compared with the one gotten from the method below), then consider the ways listed below would be practicable. Otherwise, keep using this one as it works faster. 


.. code-block:: python

  from BetabinGLM import BetaBinomial
  
  model = BetaBinomial(x, y, fit_intercept = True, method = "Nelder-Mead")
  
This package is similar to the 'betabin' mentioned above expect using a different way to initialize and calling scipy.optimize.minimize several times during optimization. Also, Nelder-Mead is used by default, instead of SLSQP. Try this one if the last can't offer a sactisfying result. 


.. code-block:: python

  from BetabinGLM import BetaBinomialAlternative
  
  model = BetaBinomialAlternative(x, y, fit_intercept = True, method = "Nelder-Mead")
  
This package is very similar to the 'BetaBinomial' mentioned above. Try this one if the data contains various variables (e.g. the number of variables > 15) since this one works faster while the accuracy are similar to the 'BetaBinomial'. 

Related links
=============

This package is an achievement by a group of summer interns through a hackathon 
project: 

* Kevin Chung, Grace Yang, Amy Ho & Jinhui Liu
* https://github.com/StatBiomed/GLM-hackathon
