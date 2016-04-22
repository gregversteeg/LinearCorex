""" Linear Total Correlation Explanation

Code below written by:
Greg Ver Steeg (gregv@isi.edu), 2015.
"""

import numpy as np
from scipy.stats import norm, rankdata  # Used for Gaussianizing data


class Corex(object):
    """
    Linear Total Correlation Explanation

    Conventions
    ----------
    Code follows sklearn naming/style (e.g. fit(X) to train).

    Parameters
    ----------
    n_hidden : int, default = 2
        The number of latent factors to use.

    max_iter : int, default = 100
        The max. number of iterations to reach convergence.

    noise : float default = 0.01
        We imagine some small fundamental measurement noise on Y. The value is arbitrary, but it sets
        the scale of the results, Y.

    verbose : int, optional
        Print verbose outputs.

    seed : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------


    References
    ----------
    [1] Greg Ver Steeg and Aram Galstyan. "The Information Sieve", 2015.
    [2] ?, Greg Ver Steeg, and Aram Galstyan. "The Information Sieve for Continuous Variables" [In progress]
    [3] Greg Ver Steeg, ?, and Aram Galstyan. "Linear Total Correlation Explanation" [In progress]
    """

    def __init__(self, n_hidden=2, max_iter=10000, noise=1., lam=0., mu=0.001, tol=0.0001, additive=True,
                 gaussianize_marginals=False, verbose=False, seed=None, copy=True, **kwargs):
        self.m = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Number of iterations to try
        self.noise = noise  # Sets the scale of Y
        self.mu = mu  # Initial step size for lagrange multiplier update
        self.tol = tol  # Threshold for convergence
        self.gaussianize_marginals = gaussianize_marginals
        self.verbose = verbose
        self.copy = copy  # Copy the data before subtracting the mean
        np.random.seed(seed)  # Set seed for deterministic results
        self.kwargs = kwargs

        # Initialize these when we fit on data
        self.n_samples, self.nv = 0, 0  # Number of samples/variables in input data
        self.ws = np.zeros((self.m, self.nv))  # m by nv array of weights
        self.additive = additive  # Whether or not to constrain to additive solutions
        self.lam = lam
        self.moments = {}  # dictionary of moments
        self.mean_x = None  # Mean is subtracted out, save for prediction/inversion
        self.updates = np.ones(self.m)  # Keep track of number of updates for each latent factor
        self.history = np.zeros((max_iter, 3))  # Keep track of values for TC, additivity and objective
        if verbose > 1:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print 'Linear CorEx with %d latent factors' % n_hidden

    @staticmethod
    def calculate_mi(moments):
        return (- 0.5 * np.log(1 - moments["X_i Y_j"] ** 2 / (moments["Y_j^2"] * moments["X_i^2"][:, np.newaxis]))).T

    @property
    def tc(self):
        """Sum_j TC(X;Y_j)"""
        return self.tcs.sum()

    def fit_transform(self, x, **kwargs):
        self.fit(x)
        return self.transform(x, **kwargs)

    def fit(self, x):
        x = np.array(x, dtype=float, copy=self.copy)
        self.n_samples, self.nv = x.shape  # Number of samples, variables in input data
        if self.gaussianize_marginals:
            x = gaussianize(x)
        else:
            self.mean_x = x.mean(axis=0)
            x -= self.mean_x
        var_x = np.einsum('li,li->i', x, x).clip(1e-10) / (self.n_samples - 1)  # Variance of x
        if self.ws.size == 0:  # Randomly initialize weights if not set
            self.ws = np.random.randn(self.m, self.nv) * self.noise ** 2 / np.sqrt(var_x)
        self.lam = self.lam * np.ones(self.nv)  # Initialize lagrange multipliers
        delta = np.inf

        for i_loop in range(self.max_iter):
            old_w = self.ws.copy()
            self._update_moments(x)  # Update moments based on w and samples, x.
            self.history[i_loop] = (self.tc, self.additivity.sum(), self.objective)  # Record convergence stats
            if self.additive and i_loop > 0:  # Update lambda dynamically to get additive solutions
                self.lam = (self.lam - self.mu * self.additivity).clip(0, 1)
            self._update_ws(i_loop)  # Update weights

            if self.additive:  # Adapt mu, the lambda step size, if necessary
                mu = 0.1 / (-np.min(self.additivity))
                if mu > self.mu and self.additivity.sum() < -0.1 * self.objective:
                    self.mu *= 1.1
                self.mu = np.clip(self.mu, 0, 10)

            delta = np.max(np.sqrt((var_x * (old_w - self.ws)**2).sum(axis=1) / (var_x * self.ws**2).sum(axis=1)))  # max relative change for each factor
            if self.verbose > 1:
                print 'TC = %0.3f\tadd = %0.3f\tobj = %0.3f\tdelta = %0.5f\t\tmu %0.4f' % \
                      (self.tc, self.additivity.sum(), self.objective, delta, self.mu)
            if delta < self.tol:  # Check for convergence
                if self.verbose:
                    print '%d iterations to tol: %f' % (i_loop, self.tol)
                break
        else:
            if self.verbose:
                print "Warning: Convergence not achieved in %d iterations. Final delta: %f" % (self.max_iter, delta)

        self.history = self.history[:i_loop + 1]
        order = np.argsort(-self.tcs)  # Largest TC components first.
        self.ws = self.ws[order]
        self._update_moments(x)  # Update moments based on w and samples, x.
        return self

    def transform(self, x, details=False):
        """Transform an array of inputs, x, into an array of k latent factors, Y.
            Optionally, you can get the remainder information and/or stop at a specified level."""
        x = np.array(x, dtype=float, copy=self.copy)
        ns, nv = x.shape
        assert self.nv == nv, "Incorrect number of variables in input, %d instead of %d" % (nv, self.nv)
        if self.gaussianize_marginals:
            x = gaussianize(x)  # Should gaussianize wrt to original data...
        else:
            x -= self.mean_x
        if details:
            moments = self._calculate_moments(x)
            return x.dot(self.ws.T), moments
        else:
            return x.dot(self.ws.T)

    def predict(self, y):
        return np.dot(self.moments["X_i Z_j"], y.T).T + self.mean_x

    def _update_moments(self, x):
        """Calculate moments based on the weights and samples. We also calculate and save MI, TC, additivity, and
        the value of the objective."""
        # TODO: I'd like to be able to calculate all these properties for transformed data.
        m = self.moments  # Dictionary of moments
        y = x.dot(self.ws.T)  # + self.noise * np.random.randn(len(x), self.m)  # Noise is included analytically
        if "X_i^2" in self.moments:
            m["X_i^2"] = self.moments["X_i^2"]
        else:
            m["X_i^2"] = np.einsum('li,li->i', x, x).clip(1e-10) / (len(x) - 1)  # Variance of x, unbiased estimate
        m["X_i Y_j"] = x.T.dot(y) / len(y)  # nv by m,  <X_i Y_j_j>
        m["cy"] = self.ws.dot(m["X_i Y_j"]) + self.noise ** 2 * np.eye(self.m)  # cov(y.T), m by m
        m["X_i Z_j"] = np.linalg.solve(m["cy"], m["X_i Y_j"].T).T
        m["X_i^2 | Y"] = (m["X_i^2"] - np.einsum('ij,ij->i', m["X_i Z_j"], m["X_i Y_j"]))
        assert np.all(m["X_i^2 | Y"] > 0), \
            "Negative expected covariance suggests numerical instability in inversion of covariance matrix for Y."
        m["Y_j^2"] = np.diag(m["cy"]).copy()
        self.mis = self.calculate_mi(m)
        mi_yj_x = 0.5 * np.log(m["Y_j^2"]) - 0.5 * np.log(self.noise ** 2)
        mi_xi_y = 0.5 * np.log(m["X_i^2"]) - 0.5 * np.log(m["X_i^2 | Y"])
        self.tcs = self.mis.sum(axis=1) - mi_yj_x
        self.additivity = self.mis.sum(axis=0) - mi_xi_y
        self.objective = self.tc - self.additivity.sum()

    def _update_ws(self, i_loop):
        """Update weights, and also the lagrange multipliers."""
        m = self.moments  # Shorthand for readability
        H = ((1. - self.lam) / m["X_i^2 | Y"] * m["X_i Z_j"].T).dot(m["X_i Z_j"])  # np.einsum('ir,i,is->rs', m["X_i Z_j"], (1. - self.lam) / m["X_i^2 | Y"], m["X_i Z_j"])
        np.fill_diagonal(H, 0)
        if self.verbose == 3:
            print H
        if np.max(np.abs(H)) > 1. / self.noise**2:
            update = dominant_subset(H, 1. / self.noise**2, self.updates)
        else:
            update = list(range(self.m))
        self.updates[update] += 1
        Q = m["X_i Y_j"].T[update] / (m["X_i^2"] * m["Y_j^2"][update, np.newaxis] - m["X_i Y_j"].T[update] ** 2)
        R = m["X_i Z_j"].T[update] / m["X_i^2 | Y"]
        S = np.dot(H[update], self.ws)
        if i_loop < 10:
            self.ws[update] = 0.5 * self.ws[update] + 0.5 * self.noise**2 * (self.lam * Q + (1 - self.lam) * R - S)
        else:
            self.ws[update] = self.noise**2 * (self.lam * Q + (1 - self.lam) * R - S)
        if self.verbose >= 2:
            print 'updating %d / %d' % (len(update), self.m), np.min(self.lam), np.max(self.lam), np.mean(self.lam)


def gaussianize(x):
    """Return an empirically gaussianized version of either 1-d or 2-d data(processed column-wise)"""
    if len(x.shape) == 1:
        return norm.ppf((rankdata(x) - 0.5) / len(x))
    return np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T


def dominant_subset(H, threshold, updates):
    """Pick a submatrix of off-diagonal matrix, H, that is diagonally dominant."""
    H = np.abs(H) / threshold
    scores = np.sum(H, axis=0)
    update = list(range(len(H)))
    while np.any(scores > 1):
        p = scores * updates[update]
        p /= p.sum()
        i = np.random.choice(len(scores), p=p)  # weighted random
        # i = np.random.choice(len(scores))  # random
        # i = np.argmax(p)  # Greedy
        update.pop(i)
        subH = H[update][:, update]
        scores = np.sum(subH, axis=0)
    # print update, inhibit
    return update
