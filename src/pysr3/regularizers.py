"""
Various regularizers (L0, LASSO, CAD, SCAD, etc)
"""

import numpy as np

from pysr3.lme.oracles import LinearLMEOracle


class Regularizer:
    """
    Template class for regularizers
    """

    def instantiate(self, **kwargs):
        """
        Attaches weights to the regularizer.

        Parameters
        ----------
        kwargs:
            whatever is needed for the regularizer to work

        Returns
        -------
        None
        """
        pass

    def forget(self):
        """
        Unlinks all problem-dependent information from the regularizer.

        Returns
        -------
        None
        """
        pass

    def value(self, x) -> float:
        """
        Returns the value for the regularizer at the point x

        Parameters
        ----------
        x: ndarray
            point

        Returns
        -------
        the value of the regularizer
        """
        pass

    def prox(self, x, alpha):
        """
        Return the value of the proximal operator evaluated at the point x and the step parameter alpha.

        Parameters
        ----------
        x: ndarray
            point.
        alpha:
            step parameter.

        Returns
        -------
        result of the application of the proximal operator to x
        """
        pass


class L0Regularizer(Regularizer):
    """
    Implements an L0-type regularizer, where the desired number of non-zero coordinates for
    fixed and random effects is given
    """

    def __init__(self,
                 nnz_tbeta=None,
                 nnz_tgamma=None,
                 independent_beta_and_gamma=False,
                 oracle: LinearLMEOracle = None):
        """
        Create the regularizer.

        Parameters
        ----------
        nnz_tbeta: int
            desired number of non-zero fixed effects
        nnz_tgamma: int
            desired number of non-zero random effects
        independent_beta_and_gamma: bool
            If true then we only can set an element of gamma as non-zero when the respective
            element of beta is non-zero too.
        oracle: LinearLMEOracle
            class that encompasses the information about the problem
        """
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma
        self.oracle = oracle
        self.independent_beta_and_gamma = independent_beta_and_gamma
        self.beta_weights = None
        self.gamma_weights = None
        self.beta_participation_in_selection = None
        self.gamma_participation_in_selection = None

    def instantiate(self, weights, **kwargs):
        """
        Attaches weights to the regularizer.

        Parameters
        ----------
        weights:
            regularization weights

        Returns
        -------
        None

        """
        beta_weights, gamma_weights = self.oracle.x_to_beta_gamma(weights)
        self.beta_weights = beta_weights
        self.gamma_weights = gamma_weights
        self.beta_participation_in_selection = beta_weights.astype(bool)
        self.gamma_participation_in_selection = gamma_weights.astype(bool)
        if self.nnz_tbeta is None:
            self.nnz_tbeta = len(beta_weights)
        if self.nnz_tgamma is None:
            self.nnz_tgamma = len(gamma_weights)

    def forget(self):
        """
        Unlinks all problem-dependent information from the regularizer.

        Returns
        -------
        None
        """
        self.beta_weights = None
        self.gamma_weights = None
        self.beta_participation_in_selection = None
        self.gamma_participation_in_selection = None

    @staticmethod
    def _take_only_k_max(x: np.ndarray, k: int):
        """
        Returns a vector b which consists of k largest elements of x (at the same places) and zeros everywhere else.

        Parameters
        ----------
        x : np.ndarray, shape = [n]
            Vector which we take largest elements from.
        k : int
            How many elements we take from x

        Returns
        -------
        b : np.ndarray, shape = [n]
            A vector which consists of k largest elements of x (at the same places) and zeros everywhere else.
        """

        b = np.zeros(len(x))
        if k > 0:
            idx_k_max = np.abs(x).argsort()[-k:]
            b[idx_k_max] = x[idx_k_max]
        return b

    def optimal_tbeta(self, beta: np.ndarray):
        """
        Returns tbeta which minimizes the loss function with all other variables fixed.

        It is a projection of beta on the sparse subspace with no more than k elements, which can be constructed by
        taking k largest elements from beta and setting the rest to be 0.

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.

        Returns
        -------
        tbeta : np.ndarray, shape = [n]
            Minimizer of the loss function w.r.t tbeta with other arguments fixed.
        """
        if self.beta_weights is not None:
            result = np.copy(beta)

            result[self.beta_participation_in_selection] = self._take_only_k_max(
                beta[self.beta_participation_in_selection],
                self.nnz_tbeta - sum(
                    ~self.beta_participation_in_selection))
            return result
        else:
            return self._take_only_k_max(beta, self.nnz_tbeta)

    def optimal_tgamma(self, tbeta, gamma):
        """
        Returns tgamma which minimizes the loss function with all other variables fixed.

        It is a projection of gamma on the sparse subspace with no more than nnz_gamma elements,
        which can be constructed by taking nnz_gamma largest elements from gamma and setting the rest to be 0.
        In addition, it preserves that for all the elements where tbeta = 0 it implies that tgamma = 0 as well.

        Parameters
        ----------
        tbeta : np.ndarray, shape = [n]
            Vector of (nnz_beta)-sparse estimates for fixed parameters.
        gamma : np.ndarray, shape = [k]
            Vector of covariance estimates of random effects.

        Returns
        -------
        tgamma : np.ndarray, shape = [k]
            Minimizer of the loss function w.r.t tgamma with other arguments fixed.
        """
        tgamma = np.copy(gamma)
        if not self.independent_beta_and_gamma:
            idx = tbeta == 0
            idx_gamma = self.oracle.beta_to_gamma_map[idx]
            idx_gamma = [i for i in (idx_gamma[idx_gamma >= 0]).astype(int) if self.gamma_participation_in_selection[i]]
            tgamma[idx_gamma] = 0

        tgamma[self.gamma_participation_in_selection] = self._take_only_k_max(
            tgamma[self.gamma_participation_in_selection],
            self.nnz_tgamma - sum(~self.gamma_participation_in_selection))
        return tgamma

    def prox(self, x, alpha):
        """
        Return the value of the proximal operator evaluated at the point x and the step parameter alpha.

        Parameters
        ----------
        x: ndarray
            point.
        alpha:
            step parameter.

        Returns
        -------
        result of the application of the proximal operator to x
        """
        beta, gamma = self.oracle.x_to_beta_gamma(x)
        tbeta = self.optimal_tbeta(beta)
        tgamma = self.optimal_tgamma(tbeta, gamma)
        return self.oracle.beta_gamma_to_x(tbeta, tgamma)

    def value(self, x):
        """
        Returns the value for the regularizer at the point x

        Parameters
        ----------
        x: ndarray
            point

        Returns
        -------
        the value of the regularizer
        """
        k = sum(x != 0)
        if k > self.nnz_tbeta + self.nnz_tgamma:
            return np.infty
        return 0


class L1Regularizer(Regularizer):
    """
    Implements an L1-regularizer, a.k.a. LASSO.
    N.B. Adaptive LASSO is implemented by providing custom weights.
    """

    def __init__(self, lam):
        """
        Creates LASSO regularizer

        Parameters
        ----------
        lam: float
            strength of the regularizer
        """
        self.lam = lam
        self.weights = None

    def instantiate(self, weights, **kwargs):
        """
        Attach regularization weights

        Parameters
        ----------
        weights: ndarray
            individual weights for the regularizer's coordinates.

        Returns
        -------
        None
        """
        self.weights = weights

    def forget(self):
        """
        Unlinks all problem-dependent information from the regularizer.

        Returns
        -------
        None
        """
        self.weights = None

    def value(self, x):
        """
        Returns the value for the regularizer at the point x

        Parameters
        ----------
        x: ndarray
            point

        Returns
        -------
        the value of the regularizer
        """
        if self.weights is not None:
            return self.weights.dot(np.abs(x))
        return self.lam * np.abs(x).sum()

    def prox(self, x, alpha):
        """
        Return the value of the proximal operator evaluated at the point x and the step parameter alpha.

        Parameters
        ----------
        x: ndarray
            point.
        alpha:
            step parameter.

        Returns
        -------
        result of the application of the proximal operator to x
        """
        if self.weights is not None:
            return (x - alpha * self.weights * self.lam).clip(0, None) \
                   - (- x - alpha * self.weights * self.lam).clip(0,
                                                                  None)
        return (x - alpha * self.lam).clip(0, None) - (- x - alpha * self.lam).clip(0, None)


class CADRegularizer(Regularizer):
    """
    Implement Clipped Absolute Deviation (CAD) regularizer
    """

    def __init__(self, rho, lam):
        """
        Creates CAD regularizer.

        Parameters
        ----------
        rho: float
            constant that prevents values larger than it from being penalized.
        lam: float
            strength of the regularizer
        """
        self.rho = rho
        self.lam = lam
        self.weights = None

    def instantiate(self, weights=None, **kwargs):
        """
        Attach regularization weights

        Parameters
        ----------
        weights: ndarray
            individual weights for the regularizer's coordinates.

        Returns
        -------
        None
        """
        self.weights = weights

    def forget(self):
        """
        Unlinks all problem-dependent information from the regularizer.

        Returns
        -------
        None
        """
        self.weights = None

    def value(self, x):
        """
        Returns the value for the regularizer at the point x

        Parameters
        ----------
        x: ndarray
            point

        Returns
        -------
        the value of the regularizer
        """
        if self.weights is not None:
            return self.lam * np.minimum(self.weights * np.abs(x), self.rho).sum()
        return self.lam * np.minimum(np.abs(x), self.rho).sum()

    def prox(self, x, alpha):
        """
        Return the value of the proximal operator evaluated at the point x and the step parameter alpha.

        Parameters
        ----------
        x: ndarray
            point.
        alpha:
            step parameter.

        Returns
        -------
        result of the application of the proximal operator to x
        """
        x = np.atleast_1d(x)
        v = np.copy(x)
        idx_small = np.where((np.abs(x) <= self.rho) & (self.weights > 0 if self.weights is not None else True))
        if self.weights is not None:
            v[idx_small] = (x[idx_small] - self.weights[idx_small] * alpha * self.lam).clip(0, None) - (
                    - x[idx_small] - self.weights[idx_small] * alpha * self.lam).clip(0,
                                                                                      None)
        else:
            v[idx_small] = (x[idx_small] - alpha * self.lam).clip(0, None) - (
                    - x[idx_small] - alpha * self.lam).clip(0, None)
        return v


class SCADRegularizer(Regularizer):
    """
    Implements Smoothly Clipped Absolute Deviations (SCAD) regularizer.
    """

    def __init__(self, rho, sigma, lam):
        """
        Creates SCAD regularizer

        Parameters
        ----------
        rho: float, rho > 1
            first knot of the spline
        sigma: float, sigma > 1
            sigma*rho is the second knot of the spline
        lam: float, lambda > 1
            strength of the regularizer
        """
        assert rho > 1
        self.rho = rho
        self.sigma = sigma
        self.lam = lam
        self.weights = None

    def instantiate(self, weights=None, **kwargs):
        """
        Attach regularization weights

        Parameters
        ----------
        weights: ndarray
            individual weights for the regularizer's coordinates.

        Returns
        -------
        None
        """

        self.weights = weights

    def forget(self):
        """
        Unlinks all problem-dependent information from the regularizer.

        Returns
        -------
        None
        """
        self.weights = None

    def value(self, x):
        """
        Returns the value for the regularizer at the point x

        Parameters
        ----------
        x: ndarray
            point

        Returns
        -------
        the value of the regularizer
        """
        total = 0
        x = np.atleast_1d(x)
        for x_i, w in zip(x, self.weights if self.weights is not None else np.ones(x.shape)):
            if abs(x_i) < self.sigma:
                total += w * self.sigma * abs(x_i)
            elif self.sigma <= abs(x_i) <= self.rho * self.sigma:
                total += w * (-x_i ** 2 + 2 * self.rho * self.sigma * abs(x_i) - self.sigma ** 2) / (2 * (self.rho - 1))
            else:
                total += w * self.sigma ** 2 * (self.rho + 1) / 2
        return self.lam * total

    def prox(self, x, alpha):
        """
        Return the value of the proximal operator evaluated at the point x and the step parameter alpha.

        Parameters
        ----------
        x: ndarray
            point.
        alpha:
            step parameter.

        Returns
        -------
        result of the application of the proximal operator to x
        """
        x = np.atleast_1d(x)
        v = np.zeros(x.shape)
        for i, w in enumerate(self.weights if self.weights is not None else np.ones(x.shape)):
            alpha_eff = alpha * self.lam * w
            if w == 0:
                v[i] = x[i]
            elif abs(x[i]) > max(self.rho, 1 + alpha_eff) * self.sigma:
                v[i] = x[i]
            elif self.sigma * (1 + alpha_eff) <= abs(x[i]) <= max(self.rho, 1 + alpha_eff) * self.sigma:
                v[i] = ((self.rho - 1) * x[i] - np.sign(x[i]) * self.rho * self.sigma * alpha_eff) / (
                        self.rho - 1 - alpha_eff)
            else:
                v[i] = (x[i] - self.sigma * alpha_eff).clip(0, None) - (- x[i] - self.sigma * alpha_eff).clip(0, None)
        return v


class DummyRegularizer(Regularizer):
    """
    Fake regularizer that has no effect.
    """

    def value(self, x):
        """
        Returns the value for the regularizer at the point x

        Parameters
        ----------
        x: ndarray
            point

        Returns
        -------
        the value of the regularizer
        """
        return 0

    def prox(self, x, alpha):
        """
        Return the value of the proximal operator evaluated at the point x and the step parameter alpha.

        Parameters
        ----------
        x: ndarray
            point.
        alpha:
            step parameter.

        Returns
        -------
        result of the application of the proximal operator to x
        """
        return x


class PositiveQuadrantRegularizer(Regularizer):

    def __init__(self, other_regularizer: Regularizer = None):
        self.other_regularizer = other_regularizer
        self.positive_coordinates = None

    def instantiate(self, weights, oracle=None, **kwargs):
        self.positive_coordinates = ([False] * oracle.problem.num_fixed_features +
                                     [True] * oracle.problem.num_random_features)
        if self.other_regularizer:
            self.other_regularizer.instantiate(weights=weights, **kwargs)

    def value(self, x):
        y = np.infty if any(x[self.positive_coordinates] < 0) else 0
        if self.other_regularizer:
            return y + self.other_regularizer.value(x)
        else:
            return y

    def prox(self, x, alpha):
        y = x.copy()
        y[self.positive_coordinates] = np.clip(x[self.positive_coordinates], 0, None)
        if self.other_regularizer:
            return self.other_regularizer.prox(y, alpha)
        else:
            return y
