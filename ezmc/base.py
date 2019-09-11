'''
Base MCMC chain and sampler classes.
Specific samplers (e.g. Metropolis) extend these classes.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sys import stdout
import multiprocessing as mp

from . import utils

class BaseChain(object):
    '''A generic MCMC chain.

    Parameters
    ----------
    n_pars : int
        Number of parameters.
    par_names : list
        (Optional) List of parameter names.
    initial_length : int
        (Optional)

    Attributes
    ----------
    iterations : int
        Starts at 0.
    jumps : int
        Number of iterations where the proposal was accepted.
    values : np.array
        Current parameter values (array of length `n_pars`).
    chain : np.array
        Chain of parameter values.
    ll : np.array
        History of log-likelihoods
    cur_ll : float
        Current log-liklihood.
    '''
    def __init__(self, n_pars, par_names=None, initial_length=int(1e+6)):
        self.n_pars = n_pars
        if par_names is None:
            par_names = ['par%i' % i for i in range(1, n_pars+1)]
        self.par_names = par_names
        self.iterations = 0
        self.jumps = 0
        self.values = np.zeros(n_pars)
        self.values[:] = np.nan
        self.chain = np.zeros((initial_length, n_pars))
        self.ll = np.zeros(initial_length)
        self.cur_ll = np.nan

    def add_sample(self, sample, ll):
        '''Append the specified parameter values and log-lik to the chain,
        and increment `self.iterations` and `self.jumps`.
        '''
        self.chain[self.iterations, :] = sample
        self.ll[self.iterations] = self.cur_ll = ll
        self.values = sample
        self.iterations += 1
        self.jumps += 1
        self._check_len()

    def reject_sample(self, old_ll):
        '''Keep the chain where it is, and increment `self.iterations`'''
        self.chain[self.iterations, :] = self.values
        self.ll[self.iterations] = self.cur_ll = old_ll
        self.iterations += 1
        self._check_len()

    def _check_len(self):
        '''Expand chains if we've run out of space.'''
        if self.iterations > self.chain.shape[0]:
            self._expand()

    def _expand(self, expand_by=int(1e+6)):
        '''Make room for more samples'''
        new_chain = np.zeros(self.interations + expand_by, self.n_pars)
        new_chain[:self.iterations] = self.chain
        self.chain = new_chain
        self.ll = utils.expand_1d_chain(self.ll, expand_by)

    def _update_results(self, burn_in=100, thin=4):
        '''Transform samples into a dataframe, and attaches it to the chain'''
        ix = self.iterations
        samples = pd.DataFrame(self.chain, columns=self.par_names)
        samples['ll'] = self.ll
        samples['iter'] = list(range(len(samples)))
        self.results = samples.iloc[burn_in:ix:thin]

    def get_results(self, burn_in=100, thin=4):
        '''Get samples after removing burn-in period and thinning.'''
        self._update_results(burn_in=burn_in, thin=thin)
        return self.results

    def get_chain(self):
        '''Get all samples, without burn-in or thinning.'''
        return self.get_results(burn_in=0, thin=1)

    def to_csv(self, filepath):
        '''Save the samples to file.
        Generally, you'll want to run `sampler.to_csv()`,
        rather than saving results from individual chains.'''
        samples = self.get_chain() # A pandas dataframe
        samples.to_csv(filepath, index=False)



class BaseSampler(object):
    '''A generic MCMC sampler.

    Parameters
    ----------
    func : function
         The log-density function for the distribution you're sampling from.
    par_names : list of str
         List of parameter names.
    noisy : Bool
        Is the density function stochastic? If `True` (default), re-evaluate it every iteration.
    visalise_func :
         A function for visualising parameter values. Not implemented yet.
    verbose :
         If > 0, print information while sampling.

    Attributes
    ----------
    func : function
         The log-density function for the distribution you're sampling from.
    par_names : list of str
         List of parameter names.
    noisy : Bool
        Is the density function stochastic? If `True` (default), re-evaluate it every iteration.
    n_pars : int
         Number of parameters
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
                 noisy=True,
                 n_chains=4,
                 visualise_func=None,
                 verbose=1):
        self.func = func
        self.par_names = par_names
        self.n_pars = len(par_names)
        self.n_chains = n_chains
        self.noisy = noisy
        self.results = None
        self.visualise_func = visualise_func
        self.verbose = verbose
        self._add_chains(self.n_chains)

    def propose(self, chain):
        '''This method is overwitten for specific samplers'''
        print('Overwrite this method for specific samplers.')

    def eval_proposal(self, proposal, chain, noisy, *args, **kwargs):
        '''This method is overwitten for specific samplers'''
        print('Overwrite this method for specific samplers. Should return old_ll, new_ll (both float), accept (bool)')

    def _add_chains(self, n_chains):
        self.chains = []
        for i in range(n_chains):
            self.chains.append(BaseChain(self.n_pars, par_names=self.par_names))

    def sample_once(self, chain_ix, *args, **kwargs):
        '''Do a single sample, on a single chain.

        Parameters
        ----------
        chain_ix : int
            Index of the chain to sample.
        *args, **kwargs : Optional arguments passed to sampler.eval_proposal()
        '''
        chain = self.chains[chain_ix]
        proposal = self.propose(chain)
        old_ll, new_ll, accept = self.eval_proposal(proposal, chain, *args, **kwargs)
        if accept:
            chain.add_sample(proposal, new_ll)
        else:
            chain.reject_sample(old_ll)

    def sample_chain(self, chain_ix, n, verbose=None, *args, **kwargs):
        ''''Draw samples on a single chain.

        Parameters
        ----------
        chain_ix : int
            Index of the chain to sample.
        n : int
            Number of samples to draw.
        verbose : int or None
            Print verbose output?
        '''
        if verbose is not None:
            self.verbose = verbose
        chain = self.chains[chain_ix]
        target_iter = chain.iterations + n
        while(chain.iterations < target_iter):
            try:
                 self.sample_once(chain_ix, *args, **kwargs)
                 if self.verbose > 0:
                     if self.verbose == 2 or chain.iterations % 20 == 0:
                         txt = '\r#%i, #jumps = %i, Pars = %s, ll = %.2f' % (
                             chain.iterations, chain.jumps,
                             ','.join(['%.4f' % v for v in chain.values]),
                             chain.cur_ll)
                         stdout.write(txt)
                         stdout.flush()
            except Exception as ex:
                raise(ex)
                # print('\nThe following exception occured:')
                # raise ex
                # print(ex)
                # print('Retrying this sample...')
        return chain

    def sample_chains(self, n=5000, parallel=False,
                      save_every=None, filepath='.ezmc_samples_ch%i.csv',
                      verbose=None, tidy=True, print_every=100,
                      *args, **kwargs):
        ''''Draw samples for all chains.

        Parameters
        ----------
        n : int
            Iterations to run.
        parallel : bool
            Run chains in parallel? For now, uses as many cores as there are chains.
        save_every: [Not implemented!] int or None
            If not None (Default), save intermediate samples to file every
            n iterations.
        filepath: [Not implemented!] str
            CSV filepath to save samples to.
        verbose : int or None
            If None (Default), use sampler settings.
        tidy: bool
            If True (Default), trunctuate chains to length of shortest chain
            before sampling.
        '''
        if save_every is not None:
            raise NotImplementedError(
                'Saving samples is not implemented yet for this class.')
        if tidy:
            self._tidy_chains()
        if not parallel:
            for i in range(len(self.chains)):
                print('\nStarting chain %i' % (i+1))
                self.sample_chain(i, n, verbose=verbose, *args, **kwargs)
        else:
            print('Sampling in parallel')
            print('Note: Progress monitoring not implemented for parallel chains yet.')
            print('Setting verbose=0')
            ## Note - Progress is lost when you interrupt parallel sampling!
            old_verbose = self.verbose
            self.verbose = 0
            import scipy
            pool = mp.Pool(self.n_chains)
            result = [pool.apply_async(utils._parallel_sample,
                                       args=(self, i, n, print_every),
                                       kwds=kwargs) for i in
                      range(self.n_chains)]
            self.chains = [r.get() for r in result]
            self.verbose = old_verbose

    def _tidy_chains(self):
        '''
        Interrupting and restarting chains can leave you with different numbers of iterations.
        Reset everything to the length of the shortest chain.
        '''
        iterations = [chain.iterations for chain in self.chains]
        n = np.min(iterations)
        for i in range(len(self.chains)):
            self.chains[i].iterations = n

    def _update_results(self, burn_in=100, thin=4):
        '''Transform samples into a dataframe, and attaches it to the sampler'''
        res = []
        for i, chain in enumerate(self.chains):
            chain._update_results(burn_in=burn_in, thin=thin)
            r = chain.results
            r['chain'] = i+1
            res.append(r)
        self.results = pd.concat(res)

    def get_results(self, burn_in=100, thin=4):
        '''Get posterior samples, after discarding burn-in samples and thinning.

        Parameters
        ---------
        burn_in: int
            How many iterations to drop at start of chains. Default 100.
        thin: int
            Proportion of iterations should be drop from chains.
            Set to 1 for no thinning. Default 4.
        '''
        self._update_results(burn_in=burn_in, thin=thin)
        return self.results

    def get_chains(self):
        '''Get all samples on all chains, without burn-in or thinning.'''
        return self.get_results(burn_in=0, thin=1)

    def to_csv(self, filename, burn_in=0, thin=1):
        '''Gets samples a dataframe (without thinning), and saves them to file.

        Equivalent to `sampler.get_results(burn_in=0, thin=1).to_csv(filename)`
        '''
        self._update_results(burn_in=burn_in, thin=thin)
        self.results.to_csv(filepath)


    def to_arviz(self, burn_in=0, thin=0):
        '''Exports posterior samples as Arviz object for visualisation.'''
        return utils.to_arviz(self, burn_in, thin)
        import arviz as az
        samples = self.get_results(burn_in=burn_in, thin=thin)
        return utils.results_to_arviz(results)
