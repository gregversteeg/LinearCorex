from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
import linear_corex as lc
from sklearn.metrics import adjusted_rand_score
import os
if not os.path.isdir('figs'):
    os.makedirs('figs')

# DATA GENERATING METHODS
def shannon_noise(c):
    """Give the noise level compatible with a certain capacity (assuming standard signal)."""
    return 1. / (np.exp(2. * c) - 1)

def random_frac(k):
    """Return a random fraction, drawn from a uniform Dirichlet (arranged from highest to lowest)."""
    return np.ones(k) / k
    #return np.sort(np.random.dirichlet(np.ones(k)))[::-1]

def gen_data_cap(p=500, n_samples=500, capacity=4., n_sources=16, correlate_sources=False):
    """Generate data. The model is that there are n_sources, Z_j, normal with unit variance.
     There are k observed vars per source, X_i = Z_j + E_i. E_i is iid normal noise with variance 'noise' (AWGN).
     Noise is chosen randomly so that the capacity for each source is fixed.
     There is a Shannon threshold relating k and noise, defined by shannon_ functions
    """
    if correlate_sources:
        eps = correlate_sources
        sources = (eps * np.random.randn(n_samples, 1) + np.random.randn(n_samples, n_sources)) / np.sqrt(1 + eps**2)
    else:
        sources = np.random.randn(n_samples, n_sources)
    k = p / n_sources
    assert p % n_sources == 0, 'For simplicity, we force k variables per source.'
    capacities = [capacity * random_frac(k) for _ in sources.T]
    noises = [shannon_noise(c) for c in capacities]
    observed = np.vstack([(source + np.sqrt(noises[j][i]) * np.random.randn(n_samples)) / np.sqrt(1 + noises[j][i])
                          for j, source in enumerate(sources.T) for i in range(k)]).T
    return observed

def group_score(groups, m, k):
    """Groups each have size k and are sequential. You only get a perfect score if everyone in the group has the same
     latent factor."""
    true_groups = [i/k for i in range(k*m)]
    return adjusted_rand_score(groups, true_groups)

ms = [4, 8, 16, 32]
ks = [4, 8, 16, 32]

Ns = [100, 1000]
Cs = [1, 4, 16]
seed = 1
gpu = False
np.random.seed(seed)
colorscheme = plt.cm.RdBu_r
fig, axs = plt.subplots(len(Ns), len(Cs), figsize=(5 * len(Cs), 5 * len(Ns)), sharex=True, sharey=True)

for Ni, N in enumerate(Ns):
    for Ci, C in enumerate(Cs):
        results = np.zeros((len(ms), len(ks)))
        for mi, m in enumerate(ms):
            for ki, k in enumerate(ks):
                print("N:{},C:{},m:{},k:{}".format(N, C, m, k))
                x = gen_data_cap(p=k * m, n_sources=m, n_samples=N, capacity=C)
                out = lc.Corex(n_hidden=m, seed=seed, verbose=1, max_iter=10000, gpu=gpu).fit(x)
                groups = np.argmax(np.abs(out.ws), axis=0)
                results[mi, ki] = group_score(groups, m, k)
        ax = axs[Ni, Ci]
        ax.set_xlabel('k')
        ax.set_ylabel('m')
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(map(str, ks))
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels(map(str, ms))
        ax.set_title("N:{},C:{}".format(N, C))
        ax.imshow(results, interpolation='none', cmap=colorscheme, vmin=-1, vmax=1)

plt.savefig("figs/accuracy.png")
plt.close('all')
