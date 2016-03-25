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

    def __init__(self, n_hidden=2, max_iter=1000, noise=0.3, tol=0.0001, additive=True,
                 gaussianize_marginals=False, verbose=False, seed=None, copy=True, **kwargs):
        self.m = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Number of iterations to try
        self.noise = noise  # Sets the scale of Y
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
        self.lam = 0
        self.mu = 0.0001  # Sets the initial step size for lagrange multipliers
        self.moments = {}  # dictionary of moments
        self.mean_x = None  # Mean is subtracted out, save for prediction/inversion
        self.tc_history = [0]  # Keep track of TC convergence for each factor
        self.o_history = [0]  # Keep track of objective
        self.add_history = [0]  # Keep track of additivity
        if verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print 'Linear CorEx with %d latent factors' % n_hidden

    # TODO: I'd like to be able to calculate all these properties for transformed data.
    @property
    def mis(self):
        """All MIs"""
        # Bias goes like -np.log1p(-1./self.n_samples)
        return self.calculate_mi(self.moments)

    @staticmethod
    def calculate_mi(moments):
        return (- 0.5 * np.log(
                1 - moments["X_i Y_j"] ** 2 / (moments["Y_j^2"] * moments["X_i^2"][:, np.newaxis]))).T

    @property
    def tcs(self):
        """TC(X;Y_j)"""
        mi_yj_x = 0.5 * np.log(self.moments["Y_j^2"]) - 0.5 * np.log(self.noise ** 2)
        return np.sum(self.mis, axis=1) - mi_yj_x

    @property
    def tc(self):
        """Sum_j TC(X;Y_j)"""
        return np.sum(self.tcs)

    @property
    def additivity(self):
        """TC(Y;X_i)."""
        mi_xi_y = 0.5 * np.log(self.moments["X_i^2"]) - 0.5 * np.log(self.moments["X_i^2 | Y"])
        return np.sum(self.mis, axis=0) - mi_xi_y

    @property
    def objective(self):
        return self.tc - np.sum(self.additivity)

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
        var_x = np.einsum('li,li->i', x, x) / (self.n_samples - 1)  # Variance of x
        self.ws = np.random.randn(self.m, self.nv) * self.noise ** 2 / np.sqrt(var_x)  # Randomly initialize weights
        self.lam = np.zeros(self.nv)  # Initialize lagrange multipliers

        for i_loop in range(self.max_iter):
            self._update_moments(x)  # Update moments based on w and samples, x.
            old_w = self.ws.copy()
            self._update_ws()
            delta = np.sqrt(((old_w - self.ws)**2).sum())
            # delta = np.mean(np.abs(self.o_history[-5:] - self.o_history[-1]))
            if self.additive:  # Update lambda dynamically to get additive solutions
                self.lam = (self.lam - self.mu * self.additivity).clip(0, 1)
            if i_loop % 10 == 9 and i_loop < self.max_iter / 10:
                if self.additivity.sum() < - 1. / self.n_samples * self.nv:
                    if (self.add_history[-1] - self.add_history[-5]) / 5. < 0.02 / self.n_samples * self.nv:
                        self.mu = min(1., self.mu * 2.)
                        # print 'update mu', self.mu

            if self.verbose:
                print 'TC = %0.3f\tadd = %0.3f\tobj = %0.3f\tdelta = %0.5f' % \
                      (self.tc, self.additivity.sum(), self.objective, delta)
            if delta < self.tol:  # Check for convergence
                if self.verbose:
                    print '%d iterations to tol: %f' % (i_loop, self.tol)
                break
        else:
            if self.verbose:
                print "Warning: Convergence was not achieved in %d iterations. Increase max_iter." % self.max_iter

        # print '%d iteration with delta: %0.5f' % (i_loop, delta)
        order = np.argsort(-self.tcs)  # Largest TC components first.
        self.ws = self.ws[order]
        self.moments = self._calculate_moments(x)  # Update moments based on w and samples, x.
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
        """Update moments and save some stats about convergence."""
        self.moments = self._calculate_moments(x)
        self.tc_history.append(self.tc)
        self.o_history.append(self.objective)
        self.add_history.append(self.additivity.sum())

    def _calculate_moments(self, x):
        """Calculate moments based on the weights and samples. Variance of X can be calculated once at the beginning and
        saved to eliminate wasted effort. (When we transform new data, though, we have to calculate variance.)"""
        m = {}  # Dictionary of moments
        y = x.dot(self.ws.T)  # + self.noise * np.random.randn(len(x), self.m)  # Noise is included analytically
        if "X_i^2" in self.moments:
            var_x = self.moments["X_i^2"]
        else:
            var_x = np.einsum('li,li->i', x, x) / (len(x) - 1)  # Variance of x
        m["X_i^2"] = var_x
        m["X_i Y_j"] = x.T.dot(y) / len(y)  # nv by m,  <X_i Y_j_j>
        m["cy"] = self.ws.dot(m["X_i Y_j"]) + self.noise ** 2 * np.eye(self.m)  # cov(y.T), m by m
        m["X_i Z_j"] = np.linalg.solve(m["cy"], m["X_i Y_j"].T).T
        m["X_i^2 | Y"] = m["X_i^2"] - np.einsum('ij,ij->i', m["X_i Z_j"], m["X_i Y_j"])
        m["Y_j^2"] = np.diag(m["cy"])
        return m

    def _update_ws(self):
        """Update weights, and also the lagrange multipliers."""
        lam = self.lam
        m = self.moments  # Shorthand for readability
        Q = m["X_i Y_j"].T / (m["X_i^2"] * m["Y_j^2"][:, np.newaxis] - (m["X_i Y_j"] ** 2).T)
        R = m["X_i Z_j"].T / m["X_i^2 | Y"]
        H = np.einsum('ir,i,is,i->rs', m["X_i Z_j"], 1. / m["X_i^2 | Y"], m["X_i Z_j"], 1. - lam)
        np.fill_diagonal(H, 0)
        S = np.dot(H, self.ws)
        self.ws = self.noise**2 * (lam * Q + (1 - lam) * R - S)
        # Alternate update rule: (no obvious benefit and requires matrix inversion! Seems less stable)
        # self.ws = np.linalg.solve(np.eye(self.m) + self.noise**2 * H, self.noise**2 * (lam * Q + (1 - lam) * R))


def gaussianize(x):
    """Return an empirically gaussianized version of either 1-d or 2-d data(processed column-wise)"""
    if len(x.shape) == 1:
        return norm.ppf((rankdata(x) - 0.5) / len(x))
    return np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T


def _sym_decorrelation(W):
    """ Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = np.linalg.eigh(np.dot(W, W.T))
    # u (resp. s) contains the eigenvectors (resp. square roots of
    # the eigenvalues) of W * W.T
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)


def q_func(c):
    # TODO: numerical accuracy? (e.g., if only one c is large)
    return np.prod(1 + c, axis=1) - 1 - np.sum(c, axis=1)
