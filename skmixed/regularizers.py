import numpy as np

from skmixed.lme.oracles import LinearLMEOracle
from skmixed.lme.problems import LinearLMEProblem


class L0Regularizer:
    def __init__(self,
                 nnz_tbeta=1,
                 nnz_tgamma=1,
                 participation_in_selection=None,
                 independent_beta_and_gamma=False,
                 oracle: LinearLMEOracle = None):
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma
        self.participation_in_selection = participation_in_selection
        self.oracle = oracle
        self.independent_beta_and_gamma = independent_beta_and_gamma

    def instantiate(self, weights):
        pass

    @staticmethod
    def _take_only_k_max(x: np.ndarray, k: int, **kwargs):
        """
        Returns a vector b which consists of k largest elements of x (at the same places) and zeros everywhere else.

        Parameters
        ----------
        x : np.ndarray, shape = [n]
            Vector which we take largest elements from.
        k : int
            How many elements we take from x
        kwargs :
            Not used, left for future and for passing debug/experimental parameters.

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

    def optimal_tbeta(self, beta: np.ndarray, **kwargs):
        """
        Returns tbeta which minimizes the loss function with all other variables fixed.

        It is a projection of beta on the sparse subspace with no more than k elements, which can be constructed by
        taking k largest elements from beta and setting the rest to be 0.

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters.

        Returns
        -------
        tbeta : np.ndarray, shape = [n]
            Minimizer of the loss function w.r.t tbeta with other arguments fixed.
        """
        if self.participation_in_selection is not None:
            result = np.copy(beta)
            result[self.participation_in_selection] = self._take_only_k_max(beta[self.participation_in_selection],
                                                                            self.nnz_tbeta - sum(
                                                                                ~self.participation_in_selection))
            return result
        else:
            return self._take_only_k_max(beta, self.nnz_tbeta, **kwargs)

    def optimal_tgamma(self, tbeta, gamma, **kwargs):
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
        kwargs :
            Not used, left for future and for passing debug/experimental parameters.

        Returns
        -------
        tgamma : np.ndarray, shape = [k]
            Minimizer of the loss function w.r.t tgamma with other arguments fixed.
        """
        tgamma = np.copy(gamma)
        if self.independent_beta_and_gamma:
            # If we don't need to set gammas to 0 whenever
            # their respective betas are zero
            return self._take_only_k_max(tgamma, self.nnz_tgamma)

        idx = tbeta == 0
        idx_gamma = self.oracle.beta_to_gamma_map[idx]
        idx_gamma = (idx_gamma[idx_gamma >= 0]).astype(int)
        tgamma[idx_gamma] = 0
        if self.participation_in_selection is not None:
            participation_idx = self.oracle.beta_to_gamma_map[self.participation_in_selection]
            participation_idx = (participation_idx[participation_idx >= 0]).astype(int)
            # if tbeta = 0 then tgamma = 0 even if this coordinate does not participate in feature selection
            not_participation_idx = self.oracle.beta_to_gamma_map[~self.participation_in_selection & (tbeta != 0)]
            not_participation_idx = (not_participation_idx[not_participation_idx >= 0]).astype(int)
            tgamma[not_participation_idx] = gamma[not_participation_idx]
            tgamma[participation_idx] = self._take_only_k_max(tgamma[participation_idx],
                                                              self.nnz_tgamma - sum(~participation_idx))
            return tgamma
        else:
            return self._take_only_k_max(tgamma, self.nnz_tgamma)

    def prox(self, x, alpha):
        beta, gamma = self.oracle.x_to_beta_gamma(x)
        tbeta = self.optimal_tbeta(beta)
        tgamma = self.optimal_tgamma(tbeta, gamma)
        return self.oracle.beta_gamma_to_x(tbeta, tgamma)

    def value(self, x):
        k = sum(x != 0)
        if k > self.nnz_tbeta + self.nnz_tgamma:
            return np.infty
        return 0


class L0Regularizer2:
    def __init__(self,
                 nnz_tbeta=1,
                 nnz_tgamma=1,
                 independent_beta_and_gamma=False,
                 participation_in_selection=None,
                 oracle: LinearLMEOracle = None):
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma
        self.oracle = oracle
        self.participation_in_selection = participation_in_selection
        self.independent_beta_and_gamma = independent_beta_and_gamma
        self.beta_weights = None
        self.gamma_weights = None

    def instantiate(self, weights):
        beta_weights, gamma_weights = self.oracle.x_to_beta_gamma(weights)
        self.beta_weights = beta_weights
        self.gamma_weights = gamma_weights
        self.beta_participation_in_selection = beta_weights.astype(bool)
        self.gamma_participation_in_selection = gamma_weights.astype(bool)

    @staticmethod
    def _take_only_k_max(x: np.ndarray, k: int, **kwargs):
        """
        Returns a vector b which consists of k largest elements of x (at the same places) and zeros everywhere else.

        Parameters
        ----------
        x : np.ndarray, shape = [n]
            Vector which we take largest elements from.
        k : int
            How many elements we take from x
        kwargs :
            Not used, left for future and for passing debug/experimental parameters.

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

    def optimal_tbeta(self, beta: np.ndarray, **kwargs):
        """
        Returns tbeta which minimizes the loss function with all other variables fixed.

        It is a projection of beta on the sparse subspace with no more than k elements, which can be constructed by
        taking k largest elements from beta and setting the rest to be 0.

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters.

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
            return self._take_only_k_max(beta, self.nnz_tbeta, **kwargs)

    def optimal_tgamma(self, tbeta, gamma, **kwargs):
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
        kwargs :
            Not used, left for future and for passing debug/experimental parameters.

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
        beta, gamma = self.oracle.x_to_beta_gamma(x)
        tbeta = self.optimal_tbeta(beta)
        tgamma = self.optimal_tgamma(tbeta, gamma)
        return self.oracle.beta_gamma_to_x(tbeta, tgamma)

    def value(self, x):
        k = sum(x != 0)
        if k > self.nnz_tbeta + self.nnz_tgamma:
            return np.infty
        return 0


class L1Regularizer:
    def __init__(self, lam, weights=None, **kwargs):
        self.lam = lam
        self.weights = weights

    def instantiate(self, weights):
        self.weights = weights

    def value(self, x):
        if self.weights is not None:
            return self.weights.dot(np.abs(x))
        return self.lam * np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        if self.weights is not None:
            return (x - alpha * self.weights * self.lam).clip(0, None) - (- x - alpha * self.weights * self.lam).clip(0,
                                                                                                                      None)
        return (x - alpha * self.lam).clip(0, None) - (- x - alpha * self.lam).clip(0, None)


class CADRegularizer:
    def __init__(self, rho, lam, weights=None, **kwargs):
        self.rho = rho
        self.lam = lam
        self.weights = weights

    def instantiate(self, weights=None):
        self.weights = weights

    def value(self, x):
        if self.weights is not None:
            return self.lam * np.minimum(self.weights * np.abs(x), self.rho).sum()
        return self.lam * np.minimum(np.abs(x), self.rho).sum()

    def prox(self, x, alpha):
        v = np.copy(x)
        idx_small = np.where((np.abs(x) <= self.rho) & (self.weights > 0 if self.weights is not None else True))
        v[idx_small] = (x[idx_small] - self.weights[idx_small] * alpha * self.lam).clip(0, None) - (
                - x[idx_small] - self.weights[idx_small] * alpha * self.lam).clip(0,
                                                                                  None)
        return v


class SCADRegularizer:
    def __init__(self, rho, sigma, lam, weights=None, **kwargs):
        assert rho > 1
        self.rho = rho
        self.sigma = sigma
        self.lam = lam
        self.weights = weights

    def instantiate(self, weights=None):
        self.weights = weights

    def value(self, x):
        total = 0
        for x_i, w in zip(x, self.weights if self.weights is not None else np.ones(x.shape)):
            if abs(x_i) < self.sigma:
                total += w * self.sigma * x_i
            elif self.sigma <= abs(x_i) <= self.rho * self.sigma:
                total += w * (-x_i ** 2 + 2 * self.rho * self.sigma * x_i - self.sigma ** 2) / (2 * (self.rho - 1))
            else:
                total += w * x_i ** 2 * (self.rho + 1) / 2
        return self.lam * total

    def prox(self, x, alpha):
        v = np.zeros(x.shape)
        for i, w in enumerate(self.weights if self.weights is not None else np.ones(x.shape)):
            alpha_eff = alpha * self.lam * w
            if w == 0:
                v[i] = x[i]
            elif abs(x[i]) > self.rho * self.sigma:
                v[i] = x[i]
            elif self.sigma * (1 + alpha_eff) <= abs(x[i]) <= self.rho * self.sigma:
                v[i] = ((self.rho - 1) * x[i] + np.sign(x[i]) * self.rho * self.sigma * alpha_eff) / (
                            self.rho - 1 - alpha_eff)
            else:
                v[i] = (x[i] - self.sigma * alpha_eff).clip(0, None) - (- x[i] - self.sigma * alpha_eff).clip(0, None)
        return v


class DummyRegularizer:
    def __init(self):
        pass

    def instantiate(self, weights=None):
        pass

    def value(self, x):
        return 0

    def prox(self, x, alpha):
        return x
