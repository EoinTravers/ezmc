
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def traceplot(samples, pars=None):
    if pars is None:
        pars = [c for c in list(samples.columns) if c not in ['iter', 'chain']]
    def plot_par(p, ax):
        plt.sca(ax)
        for ch, df in samples.groupby('chain'):
            plt.plot(df['iter'], df[p], label='Chain %i' % ch)
        nchains = len(set(samples['chain']))
        if nchains < 5:
            plt.legend()
        plt.title('Parameter: %s' % p)
    fig, axes = plt.subplots(1, len(pars), figsize=(len(pars)*6, 4))
    if len(pars) > 1:
        axes = list(axes)
        for p, ax in zip(pars, axes):
            plot_par(p, ax)
    else:
        plot_par(pars, axes)
    return fig
