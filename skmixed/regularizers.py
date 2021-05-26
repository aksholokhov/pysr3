import numpy as np

from skmixed.lme.oracles import LinearLMEOracle

class L0Regularizer:
    def __init__(self,
                 nnz_tbeta=1,
                 nnz_tgamma=1,
                 participation_in_selection=None,
                 independent_beta_and_gamma=False,
                 oracle: LinearLMEOracle = None):
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma
        self.participation_in_selection=participation_in_selection
        self.oracle = oracle
        self.independent_beta_and_gamma = independent_beta_and_gamma

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
                                                              self.nnz_tgamma - sum(~self.participation_in_selection))
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


class L1Regularizer:
    def __init__(self, lam, **kwargs):
        self.lam = lam

    def value(self, x):
        return self.lam*np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        return (x - alpha * self.lam).clip(0, None) - (- x - alpha * self.lam).clip(0, None)


class CADRegularizer:
    def __init__(self, rho, **kwargs):
        self.rho = rho

    def value(self, x):
        return np.minimum(np.abs(x), self.rho).sum()

    def prox(self, x, alpha):
        alpha = float(alpha)
        def prox_element_wise(z):
            if abs(z) > self.rho:
                return z
            elif alpha < abs(z) <= self.rho:
                return np.sign(z)*(abs(z) - alpha)
            else:
                return 0

        return np.array([prox_element_wise(z) for z in x])
