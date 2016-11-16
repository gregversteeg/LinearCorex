""" Linear Total Correlation Explanation
Now with "non-synergistic" or informative additive solutions.

Code below written by:
Greg Ver Steeg (gregv@isi.edu), 2016.
"""

import numpy as np
from scipy.stats import norm, rankdata
from scipy import linalg


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

    max_iter : int, default = 10000
        The max. number of iterations to reach convergence.

    noise : float default = 1
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
    [1] Greg Ver Steeg and Aram Galstyan. "The Information Sieve", ICML 2016.
    [2] Greg Ver Steeg, Shuyang Gao, Kyle Reing, and Aram Galstyan. "Sifting Common Information from Many Variables" [In progress]
    [3] Greg Ver Steeg, ?, and Aram Galstyan. "Linear Total Correlation Explanation [tbd]" [In progress]
    """

    def __init__(self, n_hidden=2, max_iter=10000, noise=1., lam=0., mu=0.001, tol=0.0001, additive=True,
                 gaussianize='standard', verbose=False, seed=None, **kwargs):
        self.m = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Number of iterations to try
        self.noise = noise  # Sets the scale of Y
        self.mu = mu  # Initial step size for lagrange multiplier update
        self.tol = tol  # Threshold for convergence
        self.gaussianize = gaussianize  # 'standard' translates and scale for zero mean and unit variance
        self.additive = additive  # Whether or not to constrain to additive solutions
        np.random.seed(seed)  # Set seed for deterministic results
        self.verbose = verbose
        if verbose > 1:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print 'Linear CorEx with %d latent factors' % n_hidden

        # Initialize these when we fit on data
        self.n_samples, self.nv = 0, 0  # Number of samples/variables in input data
        self.ws = np.zeros((self.m, self.nv))  # m by nv array of weights
        self.lam = lam
        self.moments = {}  # dictionary of moments
        self.theta = None  # Parameters for preprocessing marginals
        self.history = np.zeros((max_iter, 3))  # Keep track of values for TC, additivity and objective
        self.all_d = []  # Optionally, (verbose>=2) keep track of all the deltas
        self.all_tc = []  # Optionally, keep track of convergence for all tcs

    @staticmethod
    def calculate_mi(moments):
        return (- 0.5 * np.log(1 - moments["X_i Y_j"] ** 2 / (moments["Y_j^2"]))).T

    @property
    def tc(self):
        """Sum_j TC(X;Y_j)"""
        return self.tcs.sum()

    @property
    def faux(self):
        """The approximation to the objective"""
        m = self.moments
        return - 0.5 * np.sum(np.log(1 + m["Si"])) + 0.5 * np.sum(np.log(1 - m["Si"] + m["Qi"] / (1 + m["Si"]))) + \
               0.5 * np.sum(np.log(m["Y_j^2"]) - np.log(self.noise**2))

    def fit_transform(self, x, **kwargs):
        self.fit(x)
        return self.transform(x, **kwargs)

    def fit(self, x):
        x = self.preprocess(x, fit=True)  # Fit a transform for each marginal
        self.n_samples, self.nv = x.shape  # Number of samples, variables in input data
        if self.ws.size == 0:  # Randomly initialize weights if not already set
            self.ws = np.random.randn(self.m, self.nv) / np.sqrt(self.nv) * self.noise
            self.ws = _sym_decorrelation(self.ws)
        self.lam = self.lam * np.ones(self.nv)  # Initialize lagrange multipliers

        for i_loop in range(self.max_iter):
            old_w = self.ws.copy()
            self._update_moments(x)  # Update moments based on w and samples, x.
            self.history[i_loop] = (self.tc, self.additivity.sum(), self.objective)  # Record convergence stats
            if self.additive and i_loop > 0:  # Update lambda dynamically to get additive solutions
                self.lam = (self.lam - self.mu * self.additivity).clip(0, 1)

            self._update_ws_add()

            if self.additive:  # Adapt mu, the lambda step size, if necessary
                mu = 0.1 / (-np.min(self.additivity))
                if mu > self.mu and self.additivity.sum() < -0.1 * self.objective:
                    self.mu *= 1.1
                self.mu = np.clip(self.mu, 0, 10)

            deltas = np.sqrt(((old_w - self.ws)**2).sum(axis=1) / (self.ws**2).sum(axis=1))  # rel. change per factor
            if self.verbose > 1:
                print 'TC = %0.3f\tadd = %0.3f\tobj = %0.3f\tdelta = %0.5f\tfaux = %0.3f' % \
                      (self.tc, self.additivity.sum(), self.objective, deltas.sum(), self.faux)
                self.all_d.append(deltas)
                self.all_tc.append(self.tcs)
            if np.max(deltas) < self.tol:  # Check for convergence
                if self.verbose:
                    print '%d iterations to tol: %f' % (i_loop, self.tol)
                break
        else:
            if self.verbose:
                print "Warning: Convergence not achieved in %d iterations. Final delta: %f" % (self.max_iter, deltas.sum())

        self.history = self.history[:i_loop + 1]
        order = np.argsort(-self.tcs)  # Largest TC components first.
        self.ws = self.ws[order]
        self._update_moments(x)  # Update moments based on w and samples, x.
        return self

    def transform(self, x, details=False):
        """Transform an array of inputs, x, into an array of k latent factors, Y.
            Optionally, you can get the remainder information and/or stop at a specified level."""
        x = self.preprocess(x)
        ns, nv = x.shape
        assert self.nv == nv, "Incorrect number of variables in input, %d instead of %d" % (nv, self.nv)
        if details:
            raise NotImplementedError
            moments = self._calculate_moments(x)
            return x.dot(self.ws.T), moments
        else:
            return x.dot(self.ws.T)

    def preprocess(self, x, fit=False):
        """Transform each marginal to be as close to a standard Gaussian as possible. The default ('standard')
        just subtracts the mean and scales by the std. 'empirical' does an empirical gaussianization (but this cannot
        be inverted). Any other choice will skip the transformation."""
        if self.gaussianize == 'standard':
            if fit:
                mean = np.mean(x, axis=0)
                std = np.std(x, axis=0, ddof=1).clip(1e-10)
                self.theta = (mean, std)
            x = ((x - self.theta[0]) / self.theta[1])
            if np.max(np.abs(x)) > 6:
                print "Warning: outliers more than 6 stds away from mean. Consider using gaussianize='outliers'"
            return x
        elif self.gaussianize == 'outliers':
            if fit:
                mean = np.mean(x, axis=0)
                std = np.std(x, axis=0, ddof=1).clip(1e-10)
                self.theta = (mean, std)
            x = g((x - self.theta[0]) / self.theta[1])  # g truncates long tails
            return x
        elif self.gaussianize == 'empirical':
            print "Warning: correct inversion/transform of empirical gauss transform not implemented."
            return np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T
        else:
            return x

    def invert(self, x):
        """Invert the preprocessing step to get x's in the original space."""
        if self.gaussianize == 'standard':
            return self.theta[1] * x + self.theta[0]
        elif self.gaussianize == 'outliers':
            return self.theta[1] * g_inv(x) + self.theta[0]
        else:
            return x

    def predict(self, y):
        return self.invert(np.dot(self.moments["X_i Z_j"], y.T).T)

    def _update_moments(self, x):
        """Calculate moments based on the weights and samples. We also calculate and save MI, TC, additivity, and
        the value of the objective."""
        # TODO: Refactor so I can also calculate all these properties for transformed data.
        m = self.moments  # Dictionary of moments
        y = x.dot(self.ws.T)  # + self.noise * np.random.randn(len(x), self.m)  # Noise is included analytically
        m["X_i Y_j"] = x.T.dot(y) / len(y)  # nv by m,  <X_i Y_j>
        m["cy"] = self.ws.dot(m["X_i Y_j"]) + self.noise ** 2 * np.eye(self.m)  # cov(y.T), m by m
        m["X_i Z_j"] = np.linalg.solve(m["cy"], m["X_i Y_j"].T).T
        m["X_i^2 | Y"] = (1. - np.einsum('ij,ij->i', m["X_i Z_j"], m["X_i Y_j"]))
        assert np.all(m["X_i^2 | Y"] > 0), \
            "Negative expected covariance suggests numerical instability in inversion of covariance matrix for Y."
        m["Y_j^2"] = np.diag(m["cy"]).copy()
        m["ry"] = m["cy"] / (np.sqrt(m["Y_j^2"]) * np.sqrt(m["Y_j^2"][:, np.newaxis]))
        m["rho"] = (m["X_i Y_j"] / np.sqrt(m["Y_j^2"])).T
        m["invrho"] = 1. / (1. - m["rho"]**2)
        m["Qi"] = np.einsum('ji,ji,ki,ki,jk->i', m["rho"], m["invrho"], m["rho"], m["invrho"], m['ry'])
        m["Qij"] = np.einsum('ji,ji,jk->ki', m["rho"], m["invrho"], m['ry'])
        m["Si"] = np.sum(m["rho"]**2 * m["invrho"], axis=0)
        self.mis = self.calculate_mi(m)
        mi_yj_x = 0.5 * np.log(m["Y_j^2"]) - 0.5 * np.log(self.noise ** 2)
        mi_xi_y = - 0.5 * np.log(m["X_i^2 | Y"])
        self.tcs = self.mis.sum(axis=1) - mi_yj_x
        self.additivity = self.mis.sum(axis=0) - mi_xi_y
        self.objective = self.tc - self.additivity.sum()

    def _update_ws_add(self):
        """Update weights, and also the lagrange multipliers."""
        m = self.moments  # Shorthand for readability
        sy = np.sqrt(m["Y_j^2"])[:, np.newaxis]
        H = self.noise**2 * np.einsum('j,k,ji,ji,ki,ki,i->jk', 1. / np.sqrt(m["Y_j^2"]), 1. / np.sqrt(m["Y_j^2"]),
                                      m["rho"], m["invrho"], m["rho"], m["invrho"], 1. / (1 + m["Qi"] - m["Si"]**2))
        np.fill_diagonal(H, 0)
        eigs = np.linalg.eigvalsh(H)
        eta = np.clip(0.1 * (1. / np.max(np.abs(eigs))), 0, 1)
        if self.verbose == 2:
            print 'eigvals:', eigs, "eta: %0.5f" % eta
            # print H
        O1D1 = self.noise**2 * m["invrho"]**2 * m["rho"] / (1 + m["Si"]) / sy
        O1D2 = self.noise**2 / sy * m["invrho"]**2 * \
               ((1 + m["rho"]**2) * m["Qij"] - 2 * m["rho"] * m["Si"]) / (1 - m["Si"]**2 + m["Qi"]) \
               - O1D1
        # O2D2 = np.dot(H, self.ws)
        # self.ws = O1D1
        # print self.ws - (O1D1 - O1D2 - O2D2)
        # self.ws = (1 - eta) * self.ws + eta * (O1D1 - O1D2 - O2D2)
        self.ws = np.linalg.lstsq(np.eye(self.m) + H, O1D1-O1D2)[0]  # Hmmm. not robust with large m?
        #print O1D1
        #print O1D2
        #print O2D2
        #print H
        # self.ws = O1D1 - O1D2 - O2D2
        #self.ws = (1 - eta) * self.ws + eta * ws
        # if np.max(np.abs(eigs)) > 1:
        #     self.ws = np.linalg.solve(np.eye(self.m) + H, O1D1 - O1D2)
        # else:
        #     O2D2 = np.dot(H, self.ws)
        #     self.ws = O1D1 - O1D2 - O2D2
        # Qij = np.einsum('ji,ji,jk->ki', rho, invrho, m['ry'])
        # Qi = np.einsum('ji,ji,ki,ki,jk->i', rho, invrho, rho, invrho, m['ry'])
        # Ti = 1 - Si**2 + Qi / (1 + Si) # 1 - Si**2 + Qi  # 1 - Si**2 + Qi / (1 + Si) works kind of but is wrong.
        # H = self.noise ** 2 * np.einsum('ji,ji,ki,ki,j,k,i->jk', rho, invrho, rho, invrho,
        #                                 1. / np.sqrt(m["Y_j^2"]), 1. / np.sqrt(m["Y_j^2"]), 1. / Ti)
        # np.fill_diagonal(H, 0)
        # eta = np.clip(1. / (np.sum(np.abs(H), axis=0)), 0, 1)[:, np.newaxis]
        # # print H, eta.ravel()
        # ws = (2 * self.noise**2 * invrho**2 * rho / (1 + Si) / np.sqrt(m["Y_j^2"][:, np.newaxis])
        #       + self.noise**2 / np.sqrt(m["Y_j^2"][:, np.newaxis]) / Ti * invrho**2 * ((1 + rho**2) * Qi - 2 * Si * rho)
        #       + np.dot(H, self.ws))
        # self.ws = (1 - eta) * self.ws + eta * ws

    def _update_ws(self):
        """Update weights, and also the lagrange multipliers."""
        m = self.moments  # Shorthand for readability
        H = ((1. - self.lam) / m["X_i^2 | Y"] * m["X_i Z_j"].T).dot(m["X_i Z_j"])  # np.einsum('ir,i,is->rs', m["X_i Z_j"], (1. - self.lam) / m["X_i^2 | Y"], m["X_i Z_j"])
        np.fill_diagonal(H, 0)
        Q = m["X_i Y_j"].T / (m["Y_j^2"][:, np.newaxis] - m["X_i Y_j"].T ** 2)
        R = m["X_i Z_j"].T / m["X_i^2 | Y"]
        S = np.dot(H, self.ws)
        eta = np.clip(1. / ((np.sum(np.abs(H), axis=0) * 0.5 * (1 + np.random.random(self.m)) * self.noise**2)), 0, 1)[:, np.newaxis]  # damping strong competitions
        self.ws = (1. - eta) * self.ws + eta * (self.lam * Q + (1 - self.lam) * R - S)
        if self.verbose > 2:
            print 'eta', eta[:, 0]


def g(x, t=4):
    """A transformation that suppresses outliers for a standard normal."""
    xp = np.clip(x, -t, t)
    diff = np.tanh(x - xp)
    return xp + diff


def g_inv(x, t=4):
    """Inverse of g transform."""
    xp = np.clip(x, -t, t)
    diff = np.arctanh(np.clip(x - xp, -1 + 1e-10, 1 - 1e-10))
    return xp + diff


def _sym_decorrelation(W):
    """ Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = linalg.eigh(np.dot(W, W.T))
    # u (resp. s) contains the eigenvectors (resp. square roots of
    # the eigenvalues) of W * W.T
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)