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

Constants for intercept are added by default. For optimization, this package supports all methods in sp.optimize.minimize and SLSQP is used by default.

To obtain the Log-Likelihood:

.. code-block:: python

  model.LL


Related links
=============

This package is an achievement by a group of summer interns through a hackathon 
project: 

* Kevin Chung, Grace Yang, Amy Ho & Jinhui Liu
* https://github.com/StatBiomed/GLM-hackathon
