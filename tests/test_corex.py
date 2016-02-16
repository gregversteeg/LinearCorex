import numpy as np
import sys
sys.path.append('..')
import linear_corex as lc
from scipy.stats import kendalltau


kwargs = {'verbose': True, 'seed': 1}
np.random.seed(kwargs['seed'])
test_array_d = np.repeat([[0, 0, 0], [1, 1, 1]], 3, axis=0)
test_array_c = np.repeat(np.random.random((100, 1)), 3, axis=1)

# CONTINUOUS TESTS
def test_discrete():
    out = lc.Corex(n_hidden=1, **kwargs)
    y = out.fit_transform(test_array_d)
    assert np.allclose(kendalltau(y[:, 0], test_array_d[:, 0]), 1, atol=1e-3)


def test_discrete():
    out = lc.Corex(n_hidden=1, **kwargs)
    y = out.fit_transform(test_array_c)
    assert np.allclose(kendalltau(y[:, 0], test_array_c[:, 0]), 1, atol=1e-3)


def test_continuous():
    a = np.random.randn(10000, 3)
    a = np.repeat(a, 3, axis=1)
    a += 0.05 * np.random.normal(size=a.shape)
    a -= a.mean(axis=0)
    out = lc.Corex(n_hidden=3, max_iter=10, verbose=True, seed=0).fit(a)
    print out.ws
    print out.tc


def test_set(groups=(3,2), noise=0.05, samples=1000):
    a = []
    for i, k in enumerate(groups):
        a.append(np.repeat(np.random.randn(samples, 1), k, axis=1))
    a = np.hstack(a)
    a += noise * np.random.normal(size=a.shape)
    a -= a.mean(axis=0)
    return a
