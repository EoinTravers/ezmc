
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sys import stdout
from . import utils

class BaseChain(object):
    '''Generic MCMC chain'''
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
        self.cur_LL = np.nan

    def add_sample(self, sample, ll):
        self.chain[self.iterations, :] = sample
        self.ll[self.iterations] = self.cur_ll = ll
        self.values = sample
        self.iterations += 1
        self.jumps += 1
        self._check_len()

    def reject_sample(self, old_ll):
        self.chain[self.iterations, :] = self.values
        self.ll[self.iterations] = self.cur_ll = old_ll
        self.iterations += 1
        self._check_len()

    def _check_len(self):
        if self.iterations > self.chain.shape[0]:
            self._expand()

    def _expand(self, expand_by=int(1e+6)):
        new_chain = np.zeros(self.interations + expand_by, self.n_pars)
        new_chain[:self.iterations] = self.chain
        self.chain = new_chain
        self.ll = utils.expand_1d_chain(self.ll, expand_by)

    def _update_results(self, burn_in=100, thin=4):
        ix = self.iterations
        samples = pd.DataFrame(self.chain, columns=self.par_names)
        samples['ll'] = self.ll
        samples['iter'] = list(range(len(samples)))
        self.results = samples.iloc[burn_in:ix:thin]

    def get_results(self, burn_in=100, thin=4):
        self._update_results(burn_in=burn_in, thin=thin)
        return self.results

    def get_chain(self):
        return self.get_results(burn_in=0, thin=1)



class BaseSampler(object):
    '''Generic methods for all samplers'''
    def __init__(self, func, par_names,
                 visualise_func=None,
                 verbose=1):
        self.func = func
        self.par_names = par_names
        self.n_pars = len(par_names)
        self.chains = [] # Append when sampling
        self.results = None
        self.visualise_func = visualise_func
        self.verbose = verbose

    def propose(self, chain):
        print('Overwrite this method for specific samplers.')

    def eval_proposal(self, proposal, chain):
        print('Overwrite this method for specific samplers. Should return old_ll, new_ll (both float), accept (bool)')

    def add_chains(self, n_chains):
        for i in range(n_chains):
            self.chains.append(BaseChain(self.n_pars, par_names=self.par_names))

    def sample_once(self, chain_ix):
        chain = self.chains[chain_ix]
        proposal = self.propose(chain)
        old_ll, new_ll, accept = self.eval_proposal(proposal, chain)
        if accept:
            chain.add_sample(proposal, new_ll)
        else:
            chain.reject_sample(old_ll)

    def sample_chain(self, chain_ix, n, verbose=None):
        if verbose is not None:
            self.verbose = verbose
        chain = self.chains[chain_ix]
        target_iter = chain.iterations + n
        while(chain.iterations < target_iter):
            try:
                 self.sample_once(chain_ix)
                 if self.verbose > 0:
                     txt = '\r#%i, #jumps = %i, Pars = %s, ll = %.2f' % (
                         chain.iterations, chain.jumps,
                         ','.join(['%.4f' % v for v in chain.values]),
                         chain.cur_ll)
                     stdout.write(txt)
                     stdout.flush()
            except Exception as ex:
                print('\nThe following exception occured:')
                print(ex)
                print('Retrying this sample...')

    def sample_chains(self, nchains=4, n=5000, verbose=None, append=True):
        '''
        TODO:
        - Run in paralllel
        - Save progress to csv
        '''
        if append == False or len(self.chains) == 0:
            self.add_chains(nchains)
        self._tidy_chains()
        assert(len(self.chains) == nchains, 'Problem with number of chains!')
        for i in range(nchains):
            print('\nStarting chain %i' % (i+1))
            self.sample_chain(i, n, verbose=verbose)

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
        res = []
        for i, chain in enumerate(self.chains):
            chain._update_results(burn_in=burn_in, thin=thin)
            r = chain.results
            r['chain'] = i+1
            res.append(r)
        self.results = pd.concat(res)

    def get_results(self, burn_in=100, thin=4):
        self._update_results(burn_in=burn_in, thin=thin)
        return self.results

    def get_chains(self):
        return self.get_results(burn_in=0, thin=1)
