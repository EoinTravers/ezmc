# ezmc: Easy-Peasy MCMC

`ezmc` provides a simple interface to Markov Chain Monte Carlo algorithms for Bayesian inference.

It's mainly designed for models whose likelihood functions are intractable
and have to be approximated through simulations,
such as non-linear evidence accumulation models in psychology.

## Why ezmc?

MCMC is slow. To speed it up,
existing python packages such as pymc3, pystan, and emcee
rely on high-perfomance backends such as Theano or Stan.
Unfortunately, this means you need to be able to evaluate the
posterior density function in whatever backend you're using.
This isn't alway possible, in particular if you've using simulations
or other methods to approximate the density function.
`ezmc`, on the other hand, doesn't care how you evaluate posterior densities.
All it asks is that you provide a python function that takes an array of parameters as it's input,
and returns returns log posterior density as it's output.

This makes `ezmc` **simple**, easy to use, flexible, and extendable.
Unfortunately, it also makes it **slow**.
If your posterior density can be expressed analytically, this isn't the sampler for you.
It's also a work in progress, so please use at your own risk.

## Features

### Current features

- Metropolis MCMC sampler
- [Differential Evolution MCMC sampler](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4140408/)
- Export posterior samples to Pandas or Arviz.
- Traceplots

### To Do List

- Docstrings and webpage
- Metropolis
    - Run chains in parallel
    - Automatic tuning of proposal distribution
- Convergence summaries
    - Gelman-Rubin Rhat diagnostic
- Reload samples from file.
- Interoperability with other tools (we don't need to reimplement what pymc3/arviz already do)
- Unit tests.
- Optimization/Mode-finding
- Other samplers (so long as they don't require derivatives)
