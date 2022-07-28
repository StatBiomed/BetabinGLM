==========================================
BetabinGLM: Beta-binomial regression model
==========================================

Installation
============

You can install from this GitHub repository for the latest (often development) 
version by following the command line

.. code-block:: bash

  pip install -U git+https://github.com/StatBiomed/BetabinGLM

In either case, add ``--user`` if you don't have write permission for your 
Python environment.


Quick start
===========

.. code-block:: python

  from BetabinGLM import betabin

  model = betabin(x, y)

Where both x and y can be an array of more than 1 dimension

To obtain the Log-Likelihood:

.. code-block:: python

  model.LL


Related links
=============

This package is an achievement by a group of summer interns through a hackathon 
project: 

* Kevin Chung, Grace Yang, Amy Ho & Jinhui Liu
* https://github.com/StatBiomed/GLM-hackathon
