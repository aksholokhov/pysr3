from typing import Callable, List

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

from lib.problems import LinearLMEProblem


class LinearLMEOracle:
    """
    Implements Linear Mixed-Effects Model functional for given problem:
        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        u_i ~ 𝒩(0, diag(𝛄)),

        𝜺_i ~ 𝒩(0, Λ)

    The problem should be provided as LinearLMEProblem.

    """

    def __init__(self, problem: LinearLMEProblem):
        """
        Creates an oracle on top of the given problem

        Parameters
        ----------
        problem : LinearLMEProblem
            set of data and answers. See docs for LinearLMEProblem class for more details.
        """

        self.problem = problem
        self.omega_cholesky_inv = []
        self.omega_cholesky = []
        self.gamma = None

    def _recalculate_cholesky(self, gamma: np.ndarray):
        """
        Supplementary subroutine which recalculates Cholesky decompositions and their inverses of matrices Ω_i
        when 𝛄 (estimated set of covariances for random effects) changes:

        Ω_i = Z_i*diag(𝛄)*Z_i^T + Λ_i = L_i*L_i^T

        Parameters
        ----------
        gamma : np.ndarray, shape=[k]
            vector of covariances for random effects

        Returns
        -------
            None : if all the Cholesky factors were updated and stored successfully, otherwise raises and error
        """

        if (self.gamma != gamma).any():
            gamma_mat = np.diag(gamma)
            invert_upper_triangular: Callable[[np.ndarray], np.ndarray] = get_lapack_funcs("trtri")
            for x, y, z, stds in self.problem:
                omega = z.dot(gamma_mat).dot(z.T) + np.diag(stds)
                L = np.linalg.cholesky(omega)
                L_inv = invert_upper_triangular(L.T).T
                self.omega_cholesky.append(L)
                self.omega_cholesky_inv.append(L_inv)
        self.gamma = gamma
        return None

    def loss(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> float:
        """
        Returns the loss function value ℒ(β, 𝛄).

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
            result : float
                The value of the loss function: ℒ(β, 𝛄)
        """

        result = 0
        self._recalculate_cholesky(gamma)
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            result += 1 / 2 * np.sum(L_inv.dot(xi) ** 2) - np.sum(np.log(np.diag(L_inv)))
        return result

    def gradient_gamma(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns the gradient of the loss function with respect to gamma: ∇_𝛄[ℒ](β, 𝛄)

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
            grad_gamma: np.ndarray, shape = [k]
                The gradient of the loss function with respect to gamma: ∇_𝛄[ℒ](β, 𝛄)
        """

        self._recalculate_cholesky(gamma)
        grad_gamma = np.zeros(len(gamma))
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            Lz = L_inv.dot(z)
            grad_gamma += np.sum(Lz ** 2, axis=0) - Lz.T.dot(L_inv.dot(xi)) ** 2
        return grad_gamma

    def hessian_gamma(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns the Hessian of the loss function with respect to gamma ∇²_𝛄[ℒ](β, 𝛄).
        IT'S NOT IMPLEMENTED YET, it just a placeholder which raise an error

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
            hessian: np.ndarray, shape = [k, k]
                Hessian of the loss function with respect to gamma ∇²_𝛄[ℒ](β, 𝛄).
        """

        # TODO: Implement hessian gamma
        raise NotImplementedError("Hessian is not implemented yet")

    def optimal_beta(self, gamma: np.ndarray, _dont_solve_wrt_beta=False, **kwargs):
        """
        Returns beta (vector of estimations of fixed effects) which minimizes loss function for a fixed gamma.
        It's available almost exclusively in linear models. In general one should use gradient_beta and do iterative
        minimization instead.

            kernel = ∑X_i^TΩ_iX_i

            tail = ∑X_i^TΩ_iY_i

            β = (kernel)^{-1}*tail

        Parameters
        ----------
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects.
        _dont_solve_wrt_beta : bool, Optional
            If true, then it does not perform the outer matrix inversion and returns the (kernel, tail) instead.
            It's left here for the purposes of use in child classes where both the kernel and the tail should be
            adjusted to account for regularization.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        beta: np.ndarray, shape = [n]
            Vector of optimal estimates of the fixed effects for given gamma.
        """
        self._recalculate_cholesky(gamma)
        kernel = 0
        tail = 0
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            Lx = L_inv.dot(x)
            kernel += Lx.T.dot(Lx)
            tail += Lx.T.dot(L_inv.dot(y))
        if _dont_solve_wrt_beta:
            return kernel, tail
        else:
            return np.linalg.solve(kernel, tail)

    def optimal_random_effects(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns set of optimal random effects estimations for given beta and gamma.

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        u : np.ndarray, shape = [m, k]
            Set of optimal random effects estimtions for given beta and gamma

        """

        random_effects = []
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            stds_inv_mat = np.diag(1 / stds)
            # If the variance of R.E. is 0 then the R.E. is 0, so we take it into account separately
            # to keep matrices invertible.
            mask = gamma != 0
            z_masked = z[:, mask]
            gamma_masked = gamma[mask]
            u_nonzero = np.linalg.solve(np.diag(1 / gamma_masked) + z_masked.T.dot(stds_inv_mat).dot(z_masked),
                                        z.T.dot(stds_inv_mat).dot(xi)
                                        )
            u = np.zeros(len(gamma))
            u[mask] = u_nonzero
            random_effects.append(u)
        return np.array(random_effects)


class LinearLMEOracleRegularized(LinearLMEOracle):
    """
    Implements Regularized Linear Mixed-Effects Model functional for given problem:
        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        β ~ 𝒩(tb, 1/lb),

        ||tβ||_0 = nnz(β) <= nnz_tbeta,

        u_i ~ 𝒩(0, diag(𝛄)),

        𝛄 ~ 𝒩(t𝛄, 1/lg),

        ||t𝛄||_0 = nnz(t𝛄) <= nnz_tgamma,

        𝜺_i ~ 𝒩(0, Λ)

    Here tβ and t𝛄 are single variables, not multiplications (e.g. not t*β). This oracle is designed for
    a solver (LinearLMESparseModel) which searches for a sparse solution (tβ, t𝛄) with at most k and j <= k non-zero
    elements respectively. For more details, see the documentation for LinearLMESparseModel.

    The problem should be provided as LinearLMEProblem.

    """

    def __init__(self, problem: LinearLMEProblem, lb=0.1, lg=0.1, nnz_tbeta=3, nnz_tgamma=3):
        """
        Creates an oracle on top of the given problem. The problem should be in the form of LinearLMEProblem.

        Parameters
        ----------
        problem: LinearLMEProblem
            The set of data and answers. See the docs for LinearLMEProblem for more details.
        lb : float
            Regularization coefficient (inverse std) for ||β - tβ||^2
        lg : float
            Regularization coefficient (inverse std) for ||𝛄 - t𝛄||^2
        nnz_tbeta : int
            Number of non-zero elements allowed in tβ
        nnz_tgamma : int
            Number of non-zero elements allowed in t𝛄
        """

        super().__init__(problem)
        self.lb = lb
        self.lg = lg
        self.k = nnz_tbeta
        self.j = nnz_tgamma

    def optimal_beta(self, gamma: np.ndarray, tbeta: np.ndarray = None, **kwargs):
        """
        Returns beta (vector of estimations of fixed effects) which minimizes loss function for a fixed gamma.
            kernel = ∑X_i^TΩ_iX_i

            tail = ∑X_i^TΩ_iY_i

            β = (kernel + I*lb)^{-1}*(tail + lb*tbeta)

        It's available almost exclusively in linear models. In general one should use gradient_beta and do iterative
        minimization instead.

        Parameters
        ----------
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects.
        tbeta : np.ndarray, shape = [n]
            Vector of (nnz_tbeta)-sparse estimates for fixed effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        beta: np.ndarray, shape = [n]
            Vector of optimal estimates of the fixed effects for given gamma.
        """

        kernel, tail = super(self).optimal_beta(gamma, _dont_solve_wrt_beta=True, **kwargs)
        return np.linalg.solve(self.lb * np.eye(self.problem.num_features) + kernel, self.lb * tbeta + tail)

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
        kwargs

        Returns
        -------
        b : np.ndarray, shape = [n]
            A vector which consists of k largest elements of x (at the same places) and zeros everywhere else.
        """

        b = np.zeros(len(x))
        idx_k_max = x.argsort()[-k:]
        b[idx_k_max] = x[idx_k_max]
        return b

    def optimal_tbeta(self, beta: np.ndarray, **kwargs):
        """
        Returns tbeta which minimizes the loss function with all other variables fixed. It is a projection of beta
        on the sparse subspace with no more than k elements, which can be constructed by taking k largest elements from
        beta and setting the rest to be 0.

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

        return self._take_only_k_max(beta, self.k, **kwargs)

    def loss(self, beta: np.ndarray, gamma: np.ndarray, tbeta: np.ndarray = None, tgamma: np.ndarray = None, **kwargs):
        """
        Returns the loss function value ℒ(β, 𝛄) + lb/2*||β - tβ||^2 + lg/2*||𝛄 - t𝛄||^2

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects.
        tbeta : np.ndarray, shape = [n]
            Vector of (nnz_tbeta)-sparse estimates of fixed effects.
        tgamma : np.ndarray, shape = [k]
            Vector of (nnz_tgamma)-sparse estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters.

        Returns
        -------
            result : float
                The value of the loss function: ℒ(β, 𝛄) + lb/2*||β - tβ||^2 + lg/2*||𝛄 - t𝛄||^2
        """

        return (super(self).loss(beta, gamma, **kwargs)
                + self.lb / 2 * sum((beta - tbeta) ** 2)
                + self.lg / 2 * sum((gamma - tgamma) ** 2))

    def gradient_gamma(self, beta: np.ndarray, gamma: np.ndarray, tgamma: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Returns the gradient of the loss function with respect to gamma: grad_gamma =  ∇_𝛄[ℒ](β, 𝛄) + lg*(𝛄 - t𝛄)

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects.
        tgamma : np.ndarray, shape = [k]
            Vector of (nnz_tgamma)-sparse covariance estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
            grad_gamma: np.ndarray, shape = [k]
                The gradient of the loss function with respect to gamma: grad_gamma = ∇_𝛄[ℒ](β, 𝛄) + lg*(𝛄 - t𝛄)
        """

        return self.gradient_gamma(beta, gamma, **kwargs) + self.lg * (gamma - tgamma)

    def hessian_gamma(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns the Hessian of the loss function with respect to gamma: ∇²_𝛄[ℒ](β, 𝛄) + lg*I.

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
            hessian: np.ndarray, shape = [k, k]
                Hessian of the loss function with respect to gamma ∇²_𝛄[ℒ](β, 𝛄) + lg*I.
        """

        return super(self).hessian_gamma(beta, gamma, **kwargs) + self.lg * np.eye(self.problem.num_random_effects)

    def optimal_tgamma(self, tbeta, gamma, **kwargs):
        """
        Returns tgamma which minimizes the loss function with all other variables fixed. It is a projection of gamma
        on the sparse subspace with no more than nnz_gamma elements, which can be constructed by taking nnz_gamma
        largest elements from gamma and setting the rest to be 0. In addition, it preserves that for all the
        elements where tbeta = 0 it implies that tgamma = 0 as well.

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
        tgamma = np.zeros(len(gamma))
        idx = tbeta != 0
        tgamma[idx] = gamma[idx]
        return self._take_only_k_max(tgamma, self.j)
