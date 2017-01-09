# Test the ability to recover clusters of weakly correlated variables
# The weak clusters (relatively low TC) are often noisy.
# The setup is designed to mimic some of the problems we see in gene expression datasets.
# I've been using this exact setup for a while, as it tests the boundaries of methods like CorEx/sieve
# Linear Corex with synergies eliminated is the first to perfectly recover all 30 groups.
import sys
sys.path.append('..')
import linear_corex as lc
import numpy as np
import vis_corex as vc
from scipy.stats import kendalltau


verbose = 1
np.set_printoptions(precision=3, suppress=True, linewidth=200)
seed = 1
np.random.seed(seed)
colors = ['black', 'red', 'green', 'blue', 'yellow', u'indigo', u'gold', u'hotpink', u'firebrick', u'indianred',
          u'mistyrose', u'darkolivegreen', u'darkseagreen', u'pink', u'tomato', u'lightcoral', u'orangered',
          u'palegreen', u'darkslategrey', u'greenyellow', u'burlywood', u'seashell', u'mediumspringgreen',
          u'papayawhip', u'blanchedalmond', u'chartreuse', u'dimgray', u'peachpuff', u'springgreen', u'aquamarine',
          u'orange', u'lightsalmon', u'darkslategray', u'brown', u'ivory', u'dodgerblue', u'peru', u'darkgrey',
          u'lawngreen', u'chocolate', u'crimson', u'forestgreen', u'slateblue', u'cyan', u'mintcream', u'silver']

# 30 groups, 5-20 variables each, various weak correlations
# Some groups bimodal.
# Trying to mimic qualitative features of rnaseq and ADNI dataset
n_samples = 500
n_groups = 30


def standardize(s):
    return (s - np.mean(s)) / np.std(s)


def get_r(s1, s2):
    return np.mean(standardize(s1) * standardize(s2))


def observed(s):
    # Generate a randomly sized group of variables weakly correlated to
    bimodal_s = s + 0.5 * (s > 0.2).astype(float)  # occasional unbalanced, bimodal
    n = np.random.randint(3, 16)
    ns = len(s)

    output = []
    for i in range(n):
        noise_mag = np.random.choice([0.2, 0.4, 0.6])
        signal_mag = np.random.choice([0.1, 0.5, 1])
        if np.random.random() < 0.05:
            this_s = bimodal_s
        else:
            this_s = s
        output.append(signal_mag * this_s + noise_mag * np.random.randn(ns))
    return np.vstack(output).T


def score(true, predicted):
    """Compare n true signals to some number of predicted signals.
    For each true signal take the min RMSE of each predicted.
    Signals are standardized first."""
    rs = []
    assert not np.any(np.isnan(predicted)), 'nans detected'
    for t in true.T:
        rs.append(max(np.abs(kendalltau(t, p)[0]) for p in predicted.T))
    return np.array(rs)


baseline = np.random.random((n_samples, n_groups))
signal = np.random.random((n_samples, n_groups))
signal = (signal - np.mean(signal, axis=0, keepdims=True)) / np.std(signal, axis=0, keepdims=True)
data_groups = [observed(s) for s in signal.T]
order = np.argsort([-q.shape[1] for q in data_groups])
data_groups = [data_groups[i] for i in order]
signal = np.array([signal[:,i] for i in order]).T
data = np.hstack(data_groups)
print 'group sizes', map(lambda q: q.shape[1], data_groups)
print 'Data size:', data.shape

for loop_i in range(1):
    out = lc.Corex(n_hidden=n_groups, seed=seed+loop_i, verbose=verbose, max_iter=100000, tol=1e-5).fit(data)
    print 'Done, scoring:'
    scores = score(signal, out.transform(data))
    print 'TC:', out.moments["TC"]
    print 'Actual score:', scores
    print 'Number Ok, %d / %d' % (np.sum(scores > 0.5), len(scores))
    print 'total score, %0.3f' % np.sum(scores)

names = []
for j, group in enumerate(data_groups):
    color = colors[j]
    rs = map(lambda q: get_r(q, signal[:, j]), group.T)
    mis = map(lambda r: (-0.5 * np.log(1 - r**2)), rs)
    #print ','.join(map(lambda r: '%0.3f' % r, mis))
    print ('Color: %s\tNumber in group: %d\ttotal MI: %0.3f' % (color, group.shape[1], np.sum(mis))).expandtabs(30)
    for i in range(group.shape[1]):
        names.append(color + '_' + str(i))

print 'Perfect score:', score(signal, signal)
print 'Baseline score:', score(signal, baseline)
vc.vis_rep(out, data, column_label=names, prefix='weak')