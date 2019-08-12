import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sys import stdout

from . import utils
from .base import BaseSampler, BaseChain

class DEMCChain(BaseChain):
    def __init__(self, n_pars, par_names=None, initial_length=int(1e+6)):
        super().__init__(n_pars=n_pars, par_names=par_names, initial_length=initial_length)
        self.is_migration = np.zeros(initial_length)

    def _update_results(self, burn_in=100, thin=4):
        ix = self.iterations
        samples = pd.DataFrame(self.chain, columns=self.par_names)
        samples['ll'] = self.ll
        samples['iter'] = list(range(len(samples)))
        samples['is_migration'] = self.is_migration
        self.results = samples.iloc[burn_in:ix:thin]

class DifferentialEvolutionSampler(BaseSampler):
    '''A Differential Evolution Markov Chain Sampler'''
    def __init__(self, func, par_names, init_bounds,
                 n_chains=20,
                 noisy=True,
                 visualise_func=None,
                 verbose=1):
        super().__init__(func=func, par_names=par_names, n_chains=n_chains,
                         noisy=noisy,
                         visualise_func=visualise_func, verbose=verbose)
        self.init_bounds = init_bounds

    def add_chains(self, n_chains):
        self.chains = []
        for i in range(n_chains):
            self.chains.append(DEMCChain(self.n_pars, par_names=self.par_names))

    def propose(self, chain_ix, gamma='terBrack', noise=.001):
        '''
        gamma: Tuning paramater controlling how far to jump from current value.
          'terBrack': Default. 2.38 * sqrt(2*n_pars).
          'random': np.random.uniform(.5, 1)
           float: Other hard-coded value.
        noise: Uniform noise added to proposal
        '''
        chain = self.chains[chain_ix]
        if chain.iterations == 0:
            return utils.uniform_within_bounds(self.init_bounds)
        old = chain.values
        if gamma == 'terBrack':
            gamma = 2.38 * np.sqrt(2 * len(old))
        if gamma == 'random':
            gamma = np.random.uniform(.5, 1)
        ## Pick another 2 chains at random
        nch = len(self.chains)
        other_chains = np.arange(len(self.chains))
        other_chains = other_chains[other_chains != chain_ix] # But not this one!
        mum, dad = [self.chains[ix] for ix in np.random.choice(other_chains, 2)]
        innovation = gamma * (mum.values - dad.values) + np.random.uniform(-noise, noise)
        # print('Angle: %.1f deg' % np.rad2deg(np.angle( np.complex(innovation[0], innovation[1])))) ## For debugging on 2d posteriors
        ## Innovate current chain using difference between the other two.
        return old + innovation


    def eval_proposal(self, proposal, chain):
        '''Same as Metropolis?'''
        new_ll = self.func(proposal)
        if chain.iterations == 0:
            return np.nan, new_ll, True
        old = chain.values
        if self.noisy:
            old_ll = self.func(old)
        else:
            old_ll = chain.cur_ll
        p_accept = np.exp(new_ll - old_ll) ## Only if we're using Log-likelihood!
        p_accept = min(p_accept, 1)
        accept = np.random.binomial(1, p_accept)
        return old_ll, new_ll, accept

    def sample_once(self, chain_ix):
        '''Same as base?'''
        chain = self.chains[chain_ix]
        while 1:
            try:
                proposal = self.propose(chain_ix)
                old_ll, new_ll, accept = self.eval_proposal(proposal, chain)
                if accept:
                    chain.add_sample(proposal, new_ll)
                else:
                    chain.reject_sample(old_ll)
                break ## Will there be problems with iteration counts here?
            except Exception as ex:
                print('\nThe following exception occured:')
                print(ex)
                print('Retrying this sample...')

    def sample_chain(self, chain_ix, n, verbose=None):
        raise Exception('Cannot sample single chain with DEMC.')

    def sample_chains(self, n=5000,
                      migrate_every=None,
                      save_every=None, filepath='.ezmc_samples.csv',
                      verbose=None, tidy=True):
        if tidy:
            self._tidy_chains()
        if verbose is not None:
            self.verbose = verbose
        ## TODO: Run in parallel
        for i in range(n):
            if save_every is not None and  i % save_every == 0 and i > 0:
                chains = self.get_chains()
                chains.to_csv(filepath, index=False)
            if migrate_every is not None and i % migrate_every == 0 and i > 0:
                self.migration_step()
            else:
                for ch in range(len(self.chains)):
                    self.sample_once(ch)
                if self.verbose > 0:
                    jumps = np.mean([self.chains[j].jumps for j in range(self.n_chains)])
                    LL = np.array([self.chains[j].cur_ll for j in  range(self.n_chains)])
                    txt = '#%i, #jumps = %.2f, ll = %s;' % (
                                 self.chains[0].iterations, jumps,
                                 ','.join(['%.0f' % l for l in LL]))

                    txt = txt + ' ' * 80 + '\r'
                    stdout.write(txt)
                    stdout.flush()

    def migration_step(self):
        n = len(self.chains)
        sources       = np.random.choice(range(n), n, replace=False)
        destinations  = np.random.choice(range(n), n, replace=False)
        for s, d in zip(sources, destinations):
            source = self.chains[s]
            dest = self.chains[d]
            proposal = source.values + np.random.uniform(-.01, .01)
            old_ll, new_ll, accept = self.eval_proposal(proposal, dest)
            # print(s, d, source.values, dest.values, proposal, old_ll, new_ll, accept)
            dest.is_migration[dest.iterations] = 1
            if accept:
                dest.add_sample(proposal, new_ll)
            else:
                dest.reject_sample(old_ll)
