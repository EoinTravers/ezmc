import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sys import stdout

from . import utils
from .base import BaseSampler, BaseChain

class MetropolisSampler(BaseSampler):
    def __init__(self, func, par_names,
                 proposal_sd, init_func,
                 visualise_func=None,
                 verbose=1):
        super().__init__(func=func, par_names=par_names,
                                          visualise_func=visualise_func, verbose=verbose)
        self.proposal_sd = proposal_sd
        self.init_func = init_func

    def propose(self, chain):
        if chain.iterations == 0:
            return self.init_func()
        else:
            old = chain.values
            new = stats.norm.rvs(loc=old, scale=self.proposal_sd)
            return new

    def eval_proposal(self, proposal, chain):
        new_ll = self.func(proposal)
        if chain.iterations == 0:
            return np.nan, new_ll, True
        old = chain.values
        old_ll = self.func(old)
        p_accept = np.exp(new_ll - old_ll) ## Only if we're using Log-likelihood!
        p_accept = min(p_accept, 1)
        accept = np.random.binomial(1, p_accept)
        return old_ll, new_ll, accept
