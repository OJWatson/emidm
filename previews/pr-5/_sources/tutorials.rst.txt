Tutorials
=========

emidm Tutorials
---------------

These tutorials demonstrate how to use **emidm** for epidemiological modelling.

.. toctree::
   :maxdepth: 1

   notebooks/calibration
   notebooks/bayesian_inference
   notebooks/safir_inference

The **SAFIR Inference Notebook** demonstrates Bayesian inference for time-varying
R(t) using the age-structured SAFIR model with UK demographics. It simulates a
COVID-19-like epidemic matching the UK 2020 pattern and shows how to estimate
R(t) from death data using gradient-based optimization and MCMC.

Training Surrogates
-------------------

Deep learning surrogates can approximate expensive simulation models, enabling
fast parameter exploration and inference. These tutorials demonstrate how to
train neural network surrogates using data generated from emidm's SIR models.

.. toctree::
   :maxdepth: 1

   notebooks/surrogate
   notebooks/AIMS_surrogate_notebook

The **AIMS Surrogate Notebook** is a version of the surrogate training tutorial
used during the `AIMS South Africa <https://aims.ac.za/>`_ AI in Science Masters
course. It includes interactive exercises and assessment components designed for
classroom use.

General Tutorials
-----------------

Introductory material on tools and environments used in this project.

.. toctree::
   :maxdepth: 1

   notebooks/intro

