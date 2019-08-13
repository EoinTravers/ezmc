# EZMC: Easy-Peasy MCMC

`ezmc` provides a simple interface to Markov Chain Monte Carlo algorithms for Bayesian inference.

It's primarily designed to work with models whose likelihood functions are intractable
and must be approximated through simulations,
such as non-linear evidence accumulation models in psychology.

Existing python MCMC samplers such as pymc3, pystan, and emcee
achieve extremely efficient performance by
performing computations in backends such as Theano or Stan.
Unfortunately, this means it is not straightforward to use these tools
to sample from a posterior distribution estimated using model simulations.
`ezmc`, on the other hand, doesn't care how you evaluate posterior densities.
All it asks is that you provide a python function that takes an array of parameters as it's input,
and returns returns log posterior density as it's output.

This makes `ezmc` simple, easy to use, flexible, and extendable.
Unfortunately, it also makes it **slow**.
If your posterior density can be expressed analytically, this isn't the sampler for you.
It's also a work in progress, so please use at your own risk.

Current features:

- Metropolis MCMC sampler
- [Differential Evolution MCMC sampler](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4140408/)
- Export posterior samples to Pandas or Arviz.
- Traceplots

To Do List:

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
