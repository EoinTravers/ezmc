
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

def density2d_slow(x, y, nbins=50, xlim=None, ylim=None):
    if xlim is None:
        xlim = [x.min(), x.max()]
    if ylim is None:
        ylim = [y.min(), y.max()]
    kde = stats.kde.gaussian_kde([x, y])
    xi, yi = np.meshgrid(np.linspace(*xlim, num=nbins), np.linspace(*xlim, num=nbins))
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    plt.contour(xi, yi, zi)
    plt.pcolormesh(xi, yi, zi, cmap=plt.cm.viridis)

def density2d(x, y, dpi=20, xlim=None, ylim=None):
    import mpl_scatter_density
    if xlim is None:
        xlim = [x.min(), x.max()]
    if ylim is None:
        ylim = [y.min(), y.max()]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, dpi=20)

