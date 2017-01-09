from covariance import *
import numpy as np


np.random.seed(1)
prefix = 'figs/'

# Example covariance matrices
x = np.random.randn(25, 50)
cov = np.cov(x.T)
plot_cov([('True covariance', np.eye(50)), ('Data', x), ('Empirical covariance', cov)], '{}example'.format(prefix))


# EXPERIMENT 1:
p = 64
m = 8  # Number of sources to use for generating data
eps = 0.
capacity = 6
exp_prefix = '{}p={}_m={}_eps={}_c={}'.format(prefix, p, m, eps, capacity)

k_min, k_max = 3, int(np.log2(p)) + 3  # Use between 2**kmin and 2**k_max samples for train or test
ns = [2**k for k in range(k_min, k_max + 1)]
n_max = 2 * 2**k_max  # First half for training, second half for testing
x, true_cov = gen_data_cap(p=p, n_samples=n_max, n_sources=m, correlate_sources=eps, capacity=capacity)
plot_cov([('Ground Truth Covariance', true_cov)], exp_prefix)
x_train, x_test = x[:n_max / 2], x[n_max / 2:]

results = [(name, []) for name, _ in methods]
cov_grid = [[] for _ in range(len(ns))]
for imethod, (name, method) in enumerate(methods):
    print('Method: {}'.format(name))
    for ni, n in enumerate(ns):
        print('\tn={}'.format(n))
        mu = np.mean(x_train[:n], axis=0, keepdims=True)
        stds = np.std(x_train[:n], axis=0, keepdims=True, ddof=0)
        x_test_scaled = (x_train[:n] - mu) / stds
        if name == 'Ground Truth':
            cov = true_cov
        else:
            cov = method(m, x_test_scaled) * stds * stds.T
        cov_grid[ni].append(cov)
        results[imethod][1].append(score(x_test, cov))

print 'plot scores'
print results
plot_scores(ns, results, p, name=exp_prefix)

print 'animate matrices'
skip_index = [name for name, _ in methods].index("Independent")
names = [name for name, _ in methods if name is not 'Independent']
for i in range(len(cov_grid)):
    cov_grid[i] = [cov_grid[i][j] for j in range(len(cov_grid[i])) if j != skip_index]
plot_cov_grid(cov_grid, names, ns, p, exp_prefix)