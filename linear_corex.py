""" Linear Total Correlation Explanation
Recovers linear latent factors from data, like PCA/ICA/FA, etc. except that
these factors are maximally informative about relationships in the data.
We also constrain our solutions to be "non-synergistic" for better interpretability.

Code below written by:
Greg Ver Steeg (gregv@isi.edu), 2016.
"""

import numpy as np
from scipy.stats import norm, rankdata
import gc


class Corex(object):
    """
    Linear Total Correlation Explanation

    Conventions
    ----------
    Code follows sklearn naming/style (e.g. fit(X) to train, transform() to apply model to test data).

    Parameters
    ----------
    n_hidden : int, default = 2
        The number of latent factors to use.

    max_iter : int, default = 10000
        The max. number of iterations to reach convergence.

    tol : float, default = 0.0001
        Used to test for convergence.

    eliminate_synergy : bool, default = True
        Use a constraint that the information latent factors have about data is not synergistic.

    gaussianize : str, default = 'standard'
        Preprocess data so each marginal is near a standard normal. See gaussianize method for more details.

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
    [1] Greg Ver Steeg and Aram Galstyan. "Maximally Informative Hierarchical...", AISTATS 2015.
    [2] Greg Ver Steeg, Shuyang Gao, Kyle Reing, and Aram Galstyan. "Sifting Common Information from Many Variables"
    [3] Greg Ver Steeg, ?, and Aram Galstyan. "Linear Total Correlation Explanation [tbd]" [In progress]
    """

    def __init__(self, n_hidden=2, max_iter=10000, tol=0.0001, eta=0.1,
                 eliminate_synergy=True, gaussianize='standard',
                 verbose=False, noise=1., seed=None):
        self.m = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Number of iterations to try
        self.tol = tol  # Threshold for convergence
        self.eta = eta  # Number in (0,1] that controls step size in fixed point iteration.

        self.eliminate_synergy = eliminate_synergy  # Whether or not to constrain to additive solutions
        self.gaussianize = gaussianize  # Preprocess data: 'standard' scales to zero mean and unit variance

        self.noise = noise  # Can be arbitrary, but sets the scale of Y
        np.random.seed(seed)  # Set seed for deterministic results
        self.verbose = verbose
        if verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print('Linear CorEx with {:d} latent factors'.format(n_hidden))

        # Initialize these when we fit on data
        self.n_samples, self.nv = 0, 0  # Number of samples/variables in input data
        self.ws = np.zeros((self.m, self.nv))  # m by nv array of weights
        self.moments = {}  # Dictionary of moments
        self.theta = None  # Parameters for preprocessing each variable
        self.history = {}  # Keep track of values for each iteration

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def fit(self, x):
        x = self.preprocess(x, fit=True)  # Fit a transform for each marginal
        self.n_samples, self.nv = x.shape  # Number of samples, variables in input data
        if self.ws.size == 0:  # Randomly initialize weights if not already set
            self.ws = np.random.randn(self.m, self.nv) / np.sqrt(self.nv) * self.noise

        for i_loop in range(self.max_iter):
            old_w = self.ws.copy()

            # Step 1: Update moments based on w and samples, x.
            self.moments = self._calculate_moments(x)

            # Step 2: Update the weights
            if self.eliminate_synergy:
                self.ws = self._calculate_ws(self.moments)
            else:
                self.ws = self._calculate_ws_syn(self.moments)  # Older method that allows synergies

            deltas = np.sqrt(((old_w - self.ws)**2).sum(axis=1))  # abs. change per factor
            self.update_records(self.moments, deltas)  # Book-keeping
            if np.max(deltas) < self.tol:  # Check for convergence
                if self.verbose:
                    print('{:d} iterations to tol: {:f}'.format(i_loop, self.tol))
                break
        else:
            if self.verbose:
                print("Warning: Convergence not achieved in {:d} iterations. "
                      "Final delta: {:f}".format(self.max_iter, deltas.sum()))

        order = np.argsort(-self.moments["TCs"])  # Largest TC components first.
        self.ws = self.ws[order]
        self.moments = self._calculate_moments(x)  # Update moments based on sorted weights.
        return self

    def update_records(self, moments, deltas):
        """Print and store some statistics about each iteration."""
        gc.disable()  # There's a bug that slows when appending, fixed by temporarily disabling garbage collections
        self.history["TC"] = self.history.get("TC", []) + [moments["TC"]]
        self.history["additivity"] = self.history.get("additivity", []) + [moments["additivity"]]
        if self.verbose > 1:
            print("TC={:.3f}\tadd={:.3f}\tdelta={:.6f}".format(moments["TC"], moments["additivity"], deltas.sum()))
        if self.verbose:
            self.history["deltas"] = self.history.get("deltas", []) + [deltas]
            self.history["TCs"] = self.history.get("TCs", []) + [moments["TCs"]]
        gc.enable()

    @staticmethod
    def calculate_mi(moments):
        return (- 0.5 * np.log(1 - moments["X_i Y_j"] ** 2 / (moments["Y_j^2"]))).T

    @property
    def tc(self):
        return self.moments["TC"]

    @property
    def tcs(self):
        return self.moments["TCs"]

    @property
    def mis(self):
        return self.moments["MI"]

    def _calculate_moments(self, x):
        """Calculate moments based on the weights and samples. We also calculate and save MI, TC, additivity, and
        the value of the objective. Note it is assumed that <X_i^2> = 1! """
        m = {}  # Dictionary of moments
        y = x.dot(self.ws.T)  # + self.noise * np.random.randn(len(x), self.m)  # Noise is included analytically
        m["X_i Y_j"] = x.T.dot(y) / len(y)  # nv by m,  <X_i Y_j>
        m["cy"] = self.ws.dot(m["X_i Y_j"]) + self.noise ** 2 * np.eye(self.m)  # cov(y.T), m by m
        m["X_i Z_j"] = np.linalg.solve(m["cy"], m["X_i Y_j"].T).T
        m["X_i^2 | Y"] = (1. - np.einsum('ij,ij->i', m["X_i Z_j"], m["X_i Y_j"]))
        assert np.all(m["X_i^2 | Y"] > 0), \
            "Negative conditional variance suggests numerical instability in inversion of covariance matrix for Y."
        m["Y_j^2"] = np.diag(m["cy"]).copy()
        m["ry"] = m["cy"] / (np.sqrt(m["Y_j^2"]) * np.sqrt(m["Y_j^2"][:, np.newaxis]))
        m["inv"] = np.linalg.inv(m["cy"])
        m["rho"] = (m["X_i Y_j"] / np.sqrt(m["Y_j^2"])).T
        m["invrho"] = 1. / (1. - m["rho"]**2)
        m["Qij"] = np.einsum('ji,ji,jk->ki', m["rho"], m["invrho"], m['ry'])
        m["Qi"] = np.einsum('ki,ki,ki->i', m["rho"], m["invrho"], m["Qij"])
        m["Si"] = np.sum(m["rho"]**2 * m["invrho"], axis=0)
        m["MI"] = self.calculate_mi(m)
        mi_yj_x = 0.5 * np.log(m["Y_j^2"]) - 0.5 * np.log(self.noise ** 2)
        mi_xi_y = - 0.5 * np.log(m["X_i^2 | Y"])
        m["TCs"] = m["MI"].sum(axis=1) - mi_yj_x
        m["additivity"] = (m["MI"].sum(axis=0) - mi_xi_y).sum()
        # This is the objective, a lower bound for TC
        m["TC"] = 0.5 * np.sum(np.log(1 + m["Si"])) \
                         - 0.5 * np.sum(np.log(1 - m["Si"] + m["Qi"] / (1 + m["Si"]))) \
                         - 0.5 * np.sum(np.log(m["Y_j^2"]) - np.log(self.noise**2))
        return m

    def _calculate_ws(self, m):
        """Update weights, and also the lagrange multipliers.
        m is the dictionary of moments."""
        syi = 1. / np.sqrt(m["Y_j^2"])[:, np.newaxis]
        H = self.noise**2 * syi * syi.T * np.einsum('ji,ji,ki,ki,i->jk', m["rho"], m["invrho"], m["rho"], m["invrho"],
                                                    1. / (1 + m["Qi"] - m["Si"]**2))
        np.fill_diagonal(H, 0.)

        O1D1 = self.noise**2 * syi * m["invrho"]**2 * m["rho"] / (1 + m["Si"])
        O1D2 = self.noise**2 * syi * m["invrho"]**2 * \
               ((1 + m["rho"]**2) * m["Qij"] - 2 * m["rho"] * m["Si"]) / (1 - m["Si"]**2 + m["Qi"]) \
               - O1D1

        O2D2 = np.dot(H, self.ws)
        w1 = (O1D1 - O1D2 - O2D2)
        delta = w1 - self.ws
        eta = - np.einsum('ji,ji', delta, self.ws) / np.einsum('ji,ji', delta, delta)
        if self.verbose == 2:
            # print 'eigvals:', eigs
            print H
            print 'eta', eta
        if eta <= 0 or eta > 1:
            eta = self.eta  # 2. / 3.  # 2/3 appears to be special. Much worse for half, 3/4, 1/3...
        return ((1 - eta) * self.ws + eta * w1)

    def _calculate_ws_syn(self, m):
        """Update weights, without the anti-synergy constraint.
        m is the dictionary of moments."""
        H = (1. / m["X_i^2 | Y"] * m["X_i Z_j"].T).dot(m["X_i Z_j"])
        np.fill_diagonal(H, 0)
        R = m["X_i Z_j"].T / m["X_i^2 | Y"]
        S = np.dot(H, self.ws)
        # eta = np.clip(1. / ((np.sum(np.abs(H), axis=0) * 0.5 * (1 + np.random.random(self.m)) * self.noise**2)), 0, 1)[:, np.newaxis]  # damping strong competitions
        eta = 0.5
        return (1. - eta) * self.ws + eta * (R - S)

    def transform(self, x, details=False):
        """Transform an array of inputs, x, into an array of k latent factors, Y.
            Optionally, you can get the remainder information and/or stop at a specified level."""
        x = self.preprocess(x)
        ns, nv = x.shape
        assert self.nv == nv, "Incorrect number of variables in input, %d instead of %d" % (nv, self.nv)
        if details:
            moments = self._calculate_moments(x)
            return x.dot(self.ws.T), moments
        else:
            return x.dot(self.ws.T)

    def preprocess(self, x, fit=False):
        """Transform each marginal to be as close to a standard Gaussian as possible.
        'standard' (default) just subtracts the mean and scales by the std.
        'empirical' does an empirical gaussianization (but this cannot be inverted).
        'outliers' tries to squeeze in the outliers
        Any other choice will skip the transformation."""
        if self.gaussianize == 'standard':
            if fit:
                mean = np.mean(x, axis=0)
                std = np.std(x, axis=0, ddof=1).clip(1e-10)
                self.theta = (mean, std)
            x = ((x - self.theta[0]) / self.theta[1])
            if np.max(np.abs(x)) > 6 and self.verbose:
                print("Warning: outliers more than 6 stds away from mean. Consider using gaussianize='outliers'")
            return x
        elif self.gaussianize == 'outliers':
            if fit:
                mean = np.mean(x, axis=0)
                std = np.std(x, axis=0, ddof=1).clip(1e-10)
                self.theta = (mean, std)
            x = g((x - self.theta[0]) / self.theta[1])  # g truncates long tails
            return x
        elif self.gaussianize == 'empirical':
            print("Warning: correct inversion/transform of empirical gauss transform not implemented.")
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
