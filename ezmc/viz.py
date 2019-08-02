from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def traceplot(samples, pars=None):
    if pars is None:
        pars = [c for c in list(samples.columns) if c not in ['iter', 'chain']]
    fig, axes = plt.subplots(1, len(pars), figsize=(len(pars)*6, 4))
    axes = iter(axes)
    for p in pars:
        plt.sca(axes.next())
        for ch, df in samples.groupby('chain'):
            plt.plot(df['iter'], df[p], label='Chain %i' % ch)
        plt.legend()
        plt.title('Parameter: %s' % p)
    plt.show()
