from covariance import *
import numpy as np
import cPickle

np.random.seed(1)
prefix = 'figs/'

# Load covariance matrices
dates, dates_string, stocks, x = cPickle.load(open('dates,dates_string,stocks,data.dat'))
x = x - np.mean(x, axis=0, keepdims=True)
x /= np.std(x, axis=0, keepdims=True)
# x = x[:, :100]  # Subset for dev.
print("Data shape is: {}".format(x.shape))
cov = np.cov(x.T)
plot_cov([('Data', x), ('Empirical covariance', cov)], '{}_stock_raw_data'.format(prefix))

m = 60  # Number of factors for FA, CorEx
test_weeks = 26
exp_prefix = '{}stock_{}factor_{}testweeks'.format(prefix, m, test_weeks)
ns = [52 * k for k in range(1, 17)]
p = x.shape[1]
x_train, x_test = x[:-test_weeks], x[-test_weeks:]

results = [(name, []) for name, _ in methods_no_truth]
cov_grid = [[] for _ in range(len(ns))]
for imethod, (name, method) in enumerate(methods_no_truth):
    print('Method: {}'.format(name))
    for ni, n in enumerate(ns):
        print('\tn={}'.format(n))

        mu = np.mean(x_train[-n:], axis=0, keepdims=True)
        stds = np.std(x_train[-n:], axis=0, keepdims=True)
        x_test_scaled = (x_train[-n:] - mu) / stds
        cov = method(m, x_test_scaled) * stds * stds.T
        cov_grid[ni].append(cov)
        results[imethod][1].append(score(x_test, cov))

print 'plot scores'
print results
plot_scores(ns, results, p, name=exp_prefix)

print 'animate matrices'
skip_index = [name for name, _ in methods_no_truth].index("Independent")
names = [name for name, _ in methods_no_truth if name is not 'Independent']
for i in range(len(cov_grid)):
    cov_grid[i] = [cov_grid[i][j] for j in range(len(cov_grid[i])) if j != skip_index]
plot_cov_grid(cov_grid, names, ns, exp_prefix)