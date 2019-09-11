'''
The Metropolis sampler.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sys import stdout

from . import utils
from .base import BaseSampler, BaseChain

class MetropolisSampler(BaseSampler):
    '''The Metropolis MCMC sampler.

    Unless otherwise specified, properties and methods are inherited from BaseSampler.

    Parameters
    ----------
    func : function
         The log-density function for the distribution you're sampling from.
    par_names : list of str
         List of parameter names.
    proposal_sd : float or np.array of floats
         Standard deviation of the proposal distribution for each parameter.
         If only one value is given, uses the same SD for every parameter.
    n_chains : int
         Number of chains to run. Default: 4.
    noisy : Bool
        Is the density function stochastic? If `True` (default), re-evaluate it every iteration.
    visalise_func :
         A function for visualising parameter values. Not implemented yet.
    verbose :
         If > 0, print information while sampling.
         Higher numbers, more information.

    Attributes
    ----------
    func : function
         The log-density function for the distribution you're sampling from.
    par_names : list of str
         List of parameter names.
    n_pars : int
         Number of parameters
    proposal_sd : np.array of floats
         Standard deviation of the proposal distribution for each parameter.
    n_chains : int
         Number of chains
    chains :
         List of ezmc chains.
    visalise_func :
         A function for visualising parameter values. Not implemented yet.
    verbose :
         If > 0, print information while sampling.
    '''
    def __init__(self, func, par_names,
                 proposal_sd, init_func,
                 n_chains=4, noisy=True,
                 visualise_func=None,
                 verbose=1):
        super().__init__(func=func, par_names=par_names,
                         noisy=noisy,
                         n_chains=n_chains,
                         visualise_func=visualise_func, verbose=verbose)
        self.proposal_sd = proposal_sd
        self.init_func = init_func

    def propose(self, chain):
        '''Propose a new sample based on the current values.
        You shouldn't normally call this function directly when sampling,
        but it gets used by `Sampler.sample_chains()`.

        In the Metropolis sampler, proposals are normally distributed
        around the current parameters,

        .. math:: \\theta_i^* \sim N(\\theta_i, \sigma^2_i)

        where :math:`\\theta_i^*` is the proposed value of
        the i-th parameter,
        :math:`\\theta_i` is the current value,
        and :math:`\sigma^2_i` is the proposal SD for that parameter.
        '''
        if chain.iterations == 0:
            return self.init_func()
        else:
            old = chain.values
            new = stats.norm.rvs(loc=old, scale=self.proposal_sd)
            return new

    def eval_proposal(self, proposal, chain, *args, **kwargs):
        '''
        Given a proposal for this chain, evaluate the posterior density ratio
        between the proposal and the current value of the chain,
        and decide whether to accept the proposal.

        You shouldn't normally call this function directly when sampling,
        but it gets used by `Sampler.sample_chains()`.

        In Metropolis, the the probability of accepting a proposal is just

        .. math:: \\alpha = max(\\frac{P(\\theta^*)}{P(\\theta)}, 1)

        where :math:`P(\\theta)` is the posterior density for parameters :math:`\\theta`.

        Since we're using the log-density, we instead evaluate :math:`\\alpha` as

        .. math:: \\alpha = max(exp( log(P(\\theta^*)) - log(P(\\theta))), 1)
        '''
        new_ll = self.func(proposal, *args, **kwargs)
        if chain.iterations == 0:
            return np.nan, new_ll, True
        old = chain.values
        if self.noisy:
            old_ll = self.func(old, *args, **kwargs)
        else:
            old_ll = chain.cur_ll
        p_accept = np.exp(new_ll - old_ll) ## Only if we're using Log-likelihood!
        p_accept = min(p_accept, 1)
        accept = np.random.binomial(1, p_accept)
        return old_ll, new_ll, accept
