
import numpy as np
import pandas as pd
import scipy

def expand_1d_chain(chain, by):
    '''If a chain runs out of room, make it bigger.'''
    i = len(chain)
    new_chain = np.zeros(i + by)
    new_chain[:i] = chain
    return new_chain

def uniform_within_bounds(bounds):
    '''A list of uniform random numbers within the bounds.

    e.g. uniform_within_bounds([[0, 1], [10, 11]]) -> array([ 0.45340429, 10.53046333])
    '''
    bounds = np.array(bounds)
    return np.random.uniform(bounds[:, 0], bounds[:, 1])

def summarise(results, params=None, alpha=.05):
    '''Summarise a DataFrame of posterior samples.

    Parameters
    ----------
    results : pandas.DataFrame
    params : None, str, or list of str
        Parameters to summarise. If none, summarises everything prior to the 'll' column.
    alpha : float
        alpha = .05 (Default) -> 95% credible intervals.

    Returns
    -------
    out: pandas.DataFrame
        Columns reflect posterior mean, standard error, lower and
        upper confidence bounds, and posterior probability that the
        parameter is greater than 0.
    '''
    ## Think about creating a new class for posterior samples
    ## that inherits from pandas.DataFrame, but also includes
    ## methods like this.
    if params is None:
        last_col = results.columns.get_loc('ll')
        params = results.columns[:last_col].values
    samples = results[params]
    est = samples.mean()
    se = samples.std()
    a = 100*alpha*.5
    q = [a, 100 - a]
    intervals = samples.apply(lambda x: np.percentile(x, q))
    intervals.index = ['CI_low', 'CI_high']
    p_pos = samples.apply(lambda x: np.mean(x > 0))
    summary = pd.DataFrame([est, se, p_pos], index=['Estimate', 'SE', 'P(b > 0)'])
    out = pd.concat([summary, intervals]).reindex(['Estimate', 'SE', 'CI_low', 'CI_high', 'P(b > 0)']).T
    return out

def _parallel_sample(sampler, chain_ix, n, print_every=100):
    '''Top-level wrapper function used when sampling in parallel.'''
    print('\nStarting chain %i' % (chain_ix+1))
    scipy.random.seed()
    res = _parallel_sample_chain(sampler, chain_ix, n, print_every=print_every)
    # res = sampler.sample_chain(chain_ix, n)
    print('\nFinished chain %i' % (chain_ix+1))
    return res



def _parallel_sample_chain(sampler, chain_ix, n, print_every):
    ''''Proper progress monitoring for parallel chains.
    To be integrated into BaseSampler.sample_chain.
    Draw samples on a single chain.

    Parameters
    ----------
    chain_ix : int
        Index of the chain to sample.
    n : int
        Number of samples to draw.
    verbose : Ignored
    print_every: int
    '''
    chain = sampler.chains[chain_ix]
    target_iter = chain.iterations + n
    while(chain.iterations < target_iter):
        try:
             sampler.sample_once(chain_ix)
             if chain.iterations % print_every == 0:
                 txt = 'Chain %i - #%i, #jumps = %i, Pars = %s, ll = %.2f' % (
                     (chain_ix+1), chain.iterations, chain.jumps,
                     ','.join(['%.4f' % v for v in chain.values]),
                     chain.cur_ll)
                 print(txt)
        except Exception as ex:
            raise(ex)
    return chain
