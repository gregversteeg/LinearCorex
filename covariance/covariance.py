from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, GraphLassoCV
from sklearn.decomposition import FactorAnalysis
import numpy as np
import matplotlib
# matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
from scipy.stats import rankdata
from sklearn.utils.extmath import fast_logdet, pinvh
sys.path.append('..')
import linear_corex as lc
import os
if not os.path.isdir('figs'):
    os.makedirs('figs')

# COVARIANCE MATRIX ESTIMATORS
methods = [
    # (name, f(m, x))  First parameter of f is for parameters (if needed), second is data
    ("Ground Truth", lambda m, x: None),  # Handled with special logic
    ("Independent", lambda m, x: np.eye(x.shape[1])),
    ("Empirical", lambda m, x: np.cov(x.T)),
    ("Ledoit-Wolf", lambda m, x: LedoitWolf(store_precision=False, assume_centered=True, block_size=2000).fit(x).covariance_),
    ("Factor An.", lambda m, x: FactorAnalysis(n_components=m).fit(x).get_covariance()),
    ("LinCorExS", lambda m, x: lc.Corex(n_hidden=m, max_iter=10000, verbose=1, eliminate_synergy=False).fit(x).estimate_covariance()),
    ("GLASSO", lambda m, x: GraphLassoCV().fit(x).covariance_),
    ("LinCorEx", lambda m, x: lc.Corex(n_hidden=m, max_iter=10000, verbose=1, gpu=False).fit(x).estimate_covariance()),
    #("LinCorEx2", lambda m, x: lc.Corex(n_hidden=m, max_iter=10000, verbose=1, gpu=False).fit(x).estimate_covariance2())
]

methods_no_truth = [
    # (name, f(m, x))  First parameter of f is for parameters (if needed), second is data
    ("Independent", lambda m, x: np.eye(x.shape[1])),
    ("Empirical", lambda m, x: np.cov(x.T)),
    ("Ledoit-Wolf", lambda m, x: LedoitWolf(store_precision=False, assume_centered=True, block_size=2000).fit(x).covariance_),
    ("Factor An.", lambda m, x: FactorAnalysis(n_components=m).fit(x).get_covariance()),
    ("LinCorExS", lambda m, x: lc.Corex(n_hidden=m, max_iter=10000, verbose=1, eliminate_synergy=False).fit(x).estimate_covariance()),
    ("GLASSO", lambda m, x: GraphLassoCV().fit(x).covariance_),
    ("LinCorEx", lambda m, x: lc.Corex(n_hidden=m, max_iter=10000, verbose=1, gpu=False).fit(x).estimate_covariance())
]

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
    print("Correlate sources:{}".format(correlate_sources))
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
    cov = np.eye(p)
    # for j in range(n_sources):
    #     for i in range(k):
    #         for ip in range(k):
    #             if i != ip:
    #                 cov[i + k * j, ip + k * j] = 1 / np.sqrt(1 + noises[j][i]) / np.sqrt(1 + noises[j][ip])
    print 'maximum off diagonal is', 1 / np.sqrt(1 + noises[0][0]) / np.sqrt(1 + noises[0][1])
    for i in range(p):
        for ip in range(p):
            if i == ip:
                cov[i, ip] = 1
            elif i / k == ip / k:
                cov[i, ip] = 1 / np.sqrt(1 + noises[i/k][i%k]) / np.sqrt(1 + noises[i/k][ip%k])
            elif correlate_sources:
                cov[i, ip] = eps**2 / (1 + eps**2)
    return observed, cov


# SCORING METHOD
def score(test_data, cov):
    """Use negative log-likelihood to evaluate."""
    test_cov = np.cov(test_data.T)
    result = -log_likelihood(test_cov, cov) / len(test_data)  # Convert units to bits / test sample
    return np.clip(result, -1e6, 1e6)


def log_likelihood(emp_cov, model_cov):
    """Computes the sample mean of the log_likelihood under a covariance model
    computes the empirical expected log-likelihood (accounting for the
    normalization terms and scaling), allowing for universal comparison (beyond
    this software package)
    Parameters
    ----------
    emp_cov : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance
    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested
    Returns
    -------
    sample mean of the log-likelihood
    """
    p = model_cov.shape[0]
    precision = pinvh(model_cov)
    log_likelihood_ = - np.sum(emp_cov * precision) + fast_logdet(precision)
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.
    return log_likelihood_


# PLOTTING METHODS
tableau = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_scores(xs, scores, p, name='plot', option=None):
    plt.style.use('https://dl.dropboxusercontent.com/u/23723667/gv.mplstyle')
    #plt.style.use('/Users/gregv/Dropbox/Public/gv.mplstyle')
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)

    ax.yaxis.set_label_position("right")
    plt.ylabel('Negative Log Likelihood\n (avg per test sample)', fontsize=18, fontweight='bold')
    plt.xlabel('n samples (log scale)', fontsize=18, fontweight='bold')

    # Limits
    ymin = np.min([score for _, score in scores])
    ind_index = [n for n, score in scores].index('Independent')
    ymax = scores[ind_index][1][0]  # Independent is Baseline
    buffer = (ymax - ymin) / 5.
    y0, y1 = float(int((ymin - buffer)*100)) / 100., np.around(ymax + buffer, decimals=2)
    dy = 10.**int(np.log10(buffer))
    if dy < buffer / 5:
        dy *= 2
    plt.ylim(y0, y1)
    if np.allclose(np.log2(xs), np.around(np.log2(xs))):
        xs = np.log2(xs)  # Log scale
        plt.xticks(xs, ['$2^{%d}$' % x for x in xs], fontsize=16)
    x0, x1 = xs[0], xs[-1]
    plt.xlim(x0, x1)

    # Make sure your axis ticks are large enough to be easily read.
    plt.yticks(np.arange(y0, y1, dy))


    # Add bars
    # for y in np.arange(y0+1, y1, dy):
    #    plt.plot([x0, x1], [y, y], "--", lw=0.5, color="black", alpha=0.3)
    #for x in xs:
    #    plt.plot([x, x], [y0, y1], "--", lw=0.5, color="black", alpha=0.3)
    plt.plot([np.log2(p), np.log2(p)], [y0, y1], ":", lw=2., color="red", alpha=0.8)  # n=p, special color
    plt.text(np.log2(p), y0, ' $n=p$', fontsize=16, fontweight='bold', color='red')

    if option is not None:
        pass

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="off", labelright='on')

    y_pos = rankdata([errs[0] for _, errs in scores], method='ordinal').astype(float) / len(scores) * (y1 - y0) + y0
    y_pos = (np.array([np.clip(errs[0], y0, y1) for _, errs in scores]) + y_pos)/2
    for j, (method, errs) in enumerate(scores):
        print method, errs
        if method in ["Ground Truth", "Independent", "Empirical"]:
            valign = {"Ground Truth": 'top', "Independent": "bottom", "Empirical": "top"}[method]
            plt.plot(xs, errs, '-', lw=3.5, color=tableau[j])
            plt.text(xs[-1], y_pos[j], method, fontsize=18, fontweight='bold',
                     verticalalignment=valign, horizontalalignment='right', color=tableau[j])
        else:
            plt.plot(xs, errs, '-', lw=2.5, color=tableau[j])
            plt.text(xs[0], y_pos[j], method, fontsize=18, fontweight='bold',
                 verticalalignment='center', horizontalalignment='right', color=tableau[j])

        plt.savefig("{}_build{}.png".format(name, j))

    plt.tight_layout()
    plt.savefig("{}.png".format(name))
    plt.close('all')


def plot_cov(covs, name='cov'):
    plt.style.use('https://dl.dropboxusercontent.com/u/23723667/gv.mplstyle')
    # plt.style.use('/Users/gregv/Dropbox/Public/gv_heat.mplstyle')
    colorscheme = plt.cm.RdBu_r

    # Print key
    plt.figure(figsize=(5, 1))
    plt.imshow([[-1, 1]], cmap=colorscheme, interpolation='none', vmin=-1, vmax=1)
    plt.gca().set_visible(False)
    cb = plt.colorbar(orientation='horizontal', ticks=[])
    cb.outline.set_edgecolor('white')
    plt.savefig("figs/key.png")
    plt.clf()

    # Plot matrices
    plt.figure(figsize=(15, 16 * len(covs)), frameon=False)
    for i in range(len(covs)):
        ax = plt.subplot(1, len(covs), 1 + i)
        cov_name, cov = covs[i]
        ax.set_title(cov_name)
        ax.set_axis_off()
        ax.imshow(cov, interpolation='none', cmap=colorscheme, vmin=-1, vmax=1)

    plt.savefig("{}_cov.png".format(name))
    plt.close('all')

def plot_cov_grid(cov_grid, names, ns, p, name='grid'):
    colorscheme = plt.cm.RdBu_r
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # Ever using "tight layout" messes up all future animations

    # Plot matrices
    n_rows = 2
    n_cols = int(np.ceil(float(len(names)) / n_rows))
    n = cov_grid[0][0].shape[0]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5 * n_rows))
    title = fig.suptitle('', fontsize=18, fontweight='bold')

    artists = []
    for i, ax in enumerate(axs.ravel()):
        if i < len(names):
            ax.set_title(names[i])
        ax.set_axis_off()
        artist = ax.imshow(np.zeros((n, n)), interpolation='nearest', cmap=colorscheme, vmin=-1, vmax=1)
        artists.append(artist)

    def init():
        for i, artist in enumerate(artists):
            artist.set_array(np.zeros((n, n)))
        title.set_text('')
        return artists + [title]

    def animate(j):
        for i, artist in enumerate(artists):
            if i < len(names):
                artist.set_array(cov_grid[-j - 1][i])
        title.set_text('n = {} = 2^{} * p'.format(ns[-j-1], int(np.log2(float(ns[-j-1]) / p))))
        return artists + [title]

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(cov_grid), blit=True, repeat=False)
    anim.save("{}_grid.mp4".format(name), fps=0.5, extra_args=['-pix_fmt', 'yuv420p', '-vcodec', 'libx264'])
