# This code implements linear mixed-effects oracle as a subroutine for skmixed's subroutines.
# Copyright (C) 2020 Aleksei Sholokhov, aksh@uw.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import Callable

import numpy as np
import scipy as sp
from scipy import stats
from scipy.linalg.lapack import get_lapack_funcs
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

from skmixed.lme.problems import LinearLMEProblem
from skmixed.priors import NonInformativePrior
from typing import Optional


class LinearLMEOracle:
    """
    Implements Linear Mixed-Effects Model functional for given problem.

    The model is::

        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        u_i ~ 𝒩(0, diag(𝛄)),

        𝜺_i ~ 𝒩(0, Λ)

    The problem should be provided as LinearLMEProblem.

    """

    def __init__(self, problem: Optional[LinearLMEProblem], n_iter_inner=200, tol_inner=1e-6, warm_start_duals=False,
                 prior=None):
        """
        Creates an oracle on top of the given problem

        Parameters
        ----------
        problem : LinearLMEProblem
            set of data and answers. See docs for LinearLMEProblem class for more details.
        """
        self.problem = problem
        self.prior = prior if prior else NonInformativePrior()
        self.beta_to_gamma_map = None
        self.omega_cholesky_inv = []
        self.omega_cholesky = []
        self.gamma = None
        self.n_iter_inner = n_iter_inner
        self.tol_inner = tol_inner
        if warm_start_duals:
            self.v = None
        if problem:
            self.instantiate(problem)

    def instantiate(self, problem):
        self.problem = problem
        beta_to_gamma_map = np.zeros(self.problem.num_fixed_effects)
        beta_counter = 0
        gamma_counter = 0
        for label in self.problem.column_labels:
            if label == 1:
                beta_to_gamma_map[beta_counter] = -1
                beta_counter += 1
            elif label == 2:
                gamma_counter += 1
            elif label == 3:
                beta_to_gamma_map[beta_counter] = gamma_counter
                beta_counter += 1
                gamma_counter += 1
            else:
                continue
        self.beta_to_gamma_map = beta_to_gamma_map
        self.prior.instantiate(problem)

    def forget(self):
        self.problem = None
        self.beta_to_gamma_map = None
        self.omega_cholesky_inv = []
        self.omega_cholesky = []
        self.gamma = None
        self.prior.forget()

    def _recalculate_cholesky(self, gamma: np.ndarray):
        """
        Recalculates Cholesky factors of all Ω_i's when gamma changes.

        Supplementary subroutine which recalculates Cholesky decompositions and their inverses of matrices Ω_i
        when 𝛄 (estimated set of covariances for random effects) changes:

        Ω_i = Z_i*diag(𝛄)*Z_i^T + Λ_i = L_i*L_i^T

        Parameters
        ----------
        gamma : np.ndarray, shape=[k]
            vector of covariances for random effects

        Returns
        -------
            None :
                if all the Cholesky factors were updated and stored successfully, otherwise raises and error
        """

        if (self.gamma != gamma).any():
            self.omega_cholesky = []
            self.omega_cholesky_inv = []
            gamma_mat = np.diag(gamma)
            invert_upper_triangular: Callable[[np.ndarray], np.ndarray] = get_lapack_funcs("trtri")
            for x, y, z, stds in self.problem:
                omega = z.dot(gamma_mat).dot(z.T) + np.diag(stds)
                L = np.linalg.cholesky(omega)
                L_inv = invert_upper_triangular(L.T)[0].T
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
        return result + self.prior.loss(beta, gamma)

    def demarginalized_loss(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> float:
        result = 0
        self._recalculate_cholesky(gamma)
        us = self.optimal_random_effects(beta, gamma)
        for (x, y, z, stds), u in zip(self.problem, us):
            r = y - x.dot(beta) - z.dot(u)
            result += 1 / 2 * sum(r ** 2 / stds) + 1 / 2 * sum(np.log(stds))
        return result + self.prior.loss(beta, gamma)

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
            grad_gamma += 1 / 2 * np.sum(Lz ** 2, axis=0) - 1 / 2 * Lz.T.dot(L_inv.dot(xi)) ** 2
        return grad_gamma + self.prior.gradient_gamma(beta, gamma)

    def hessian_gamma(self, beta: np.ndarray, gamma: np.ndarray, take_only_positive_part=False, **kwargs) -> np.ndarray:
        """
        Returns the Hessian of the loss function with respect to gamma ∇²_𝛄[ℒ](β, 𝛄).

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
        self._recalculate_cholesky(gamma)
        num_random_effects = self.problem.num_random_effects
        hessian = np.zeros(shape=(num_random_effects, num_random_effects))
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            Lz = L_inv.dot(z)
            Lxi = L_inv.dot(xi).reshape((len(xi), 1))
            hessian += ((0 if take_only_positive_part else -Lz.T.dot(Lz)) + 2 * (Lz.T.dot(Lxi).dot(Lxi.T).dot(Lz))) * (
                Lz.T.dot(Lz))
        return 1 / 2 * hessian + self.prior.hessian_gamma(beta, gamma)

    def optimal_gamma_pgd(self, beta: np.ndarray, gamma: np.ndarray, log_progress=False, **kwargs):
        step_len = 1
        iteration = 0
        direction = -self.gradient_gamma(beta, gamma, **kwargs)
        # projecting the direction onto the constraints (positive box for gamma)
        direction[(direction < 0) & (gamma == 0.0)] = 0
        if log_progress:
            self.logger = [gamma]
        while step_len > 0 and iteration < self.n_iter_inner and np.linalg.norm(direction) > self.tol_inner:
            ind_neg_dir = np.where(direction < 0.0)[0]
            max_step_len = min(1, 1 if len(ind_neg_dir) == 0 else np.min(-gamma[ind_neg_dir] / direction[ind_neg_dir]))
            res = sp.optimize.minimize(
                fun=lambda a: self.loss(beta, gamma + a * direction, **kwargs),
                x0=np.array([0]),
                method="TNC",
                jac=lambda a: direction.dot(self.gradient_gamma(beta, gamma + a * direction, **kwargs)),
                bounds=[(0, max_step_len)]
            )
            step_len = res.x
            gamma = gamma + step_len * direction
            gamma[gamma <= 1e-18] = 0  # killing effective zeros
            iteration += 1
            direction = -self.gradient_gamma(beta, gamma, **kwargs)
            # projecting the direction onto the constraints (positive box for gamma)
            direction[(direction < 0) & (gamma == 0.0)] = 0
            if log_progress:
                self.logger.append(gamma)
        return gamma

    def optimal_gamma_ip(self, beta: np.ndarray, gamma: np.ndarray, log_progress=False, **kwargs):
        n = len(gamma)
        I = np.eye(n)
        # v = x[:n], g = x[n:]
        F_coord = lambda v, g, mu: np.concatenate([
            v * g - mu,
            self.gradient_gamma(beta, g, **kwargs) - v
        ])
        F = lambda x, mu: F_coord(x[:n], x[n:], mu)
        dF_coord = lambda v, g: np.block([
            [np.diag(g), np.diag(v)],
            [-I, self.hessian_gamma(beta, g, take_only_positive_part=True, **kwargs)]
        ])
        dF = lambda x: dF_coord(x[:n], x[n:])
        v = np.ones(n)
        g = gamma
        x = np.concatenate([v, g])
        mu = v.dot(gamma) / n
        step_len = 1
        iteration = 0
        if log_progress:
            self.logger = [gamma]
        losses = []
        losses_kkt = []
        while step_len != 0 and iteration < self.n_iter_inner and np.linalg.norm(F(x, mu)) > self.tol_inner:
            F_current = F(x, mu)
            dF_current = dF(x)
            direction = np.linalg.solve(dF_current, -F_current)
            direction[(x == 0.0) & (direction < 0.0)] = 0
            ind_neg_dir = np.where(direction < 0.0)[0]
            max_step_len = min(1, 1 if len(ind_neg_dir) == 0 else np.min(-x[ind_neg_dir] / direction[ind_neg_dir]))
            res = sp.optimize.minimize(fun=lambda alpha: np.linalg.norm(F(x + alpha * direction, mu)) ** 2,
                                       x0=np.array([max_step_len]),
                                       method="TNC",
                                       jac=lambda alpha: 2 * F(x + alpha * direction, mu).dot(
                                           dF(x + alpha * direction).dot(direction)),
                                       bounds=[(0, max_step_len)])
            step_len = res.x
            x = x + step_len * direction
            x[x <= 1e-18] = 0  # killing effective zeros
            mu = 0.1 * x[:n].dot(x[n:]) / n
            iteration += 1
            losses.append(self.loss(beta, x[n:], **kwargs))
            losses_kkt.append(np.linalg.norm(F(x, mu)))
            if log_progress:
                self.logger.append(x[n:])
        return x[n:]

    def optimal_gamma(self, beta: np.ndarray, gamma: np.ndarray, method="pgd", **kwargs) -> np.ndarray:
        if method == "pgd":
            return self.optimal_gamma_pgd(beta, gamma, **kwargs)
        elif method == "ip":
            return self.optimal_gamma_ip(beta, gamma, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def gradient_beta(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        self._recalculate_cholesky(gamma)
        gradient = np.zeros(self.problem.num_fixed_effects)
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            gradient += - (L_inv.dot(x)).T.dot(L_inv.dot(xi))
        return gradient + self.prior.gradient_beta(beta=beta, gamma=gamma)

    def hessian_beta(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        self._recalculate_cholesky(gamma)
        hessian = np.zeros((self.problem.num_fixed_effects, self.problem.num_fixed_effects))
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            Lx = L_inv.dot(x)
            hessian += Lx.T.dot(Lx)
        return hessian + self.prior.hessian_beta(beta=beta, gamma=gamma)

    def flip_probabilities_beta(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        hessian = self.hessian_beta(beta=beta, gamma=gamma)
        cov_matrix = np.linalg.inv(hessian)
        probabilities = []
        for mean, var, sign in zip(beta, np.diag(cov_matrix),  np.sign(beta)):
            if sign > 0:
                probabilities.append(stats.norm.cdf(0, loc=mean, scale=np.sqrt(var)))
            elif sign < 0:
                probabilities.append(1 - stats.norm.cdf(0, loc=mean, scale=np.sqrt(var)))
            else:
                probabilities.append(0.5)
        return np.array(probabilities)


    def hessian_beta_gamma(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        self._recalculate_cholesky(gamma)
        hessian = np.zeros((self.problem.num_random_effects, self.problem.num_fixed_effects))
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            Lx = L_inv.dot(x)
            Lz = L_inv.dot(z)
            Lxi = L_inv.dot(xi)
            hessian += np.diag(Lz.T.dot(Lxi)).dot(Lz.T.dot(Lx))
        return hessian.T + self.prior.hessian_beta_gamma(beta=beta, gamma=gamma)

    def x_to_beta_gamma(self, x):
        beta = x[:self.problem.num_fixed_effects]
        gamma = x[self.problem.num_fixed_effects:self.problem.num_fixed_effects + self.problem.num_random_effects]
        return beta, gamma

    def beta_gamma_to_x(self, beta, gamma):
        return np.concatenate([beta, gamma])

    def joint_loss(self, x, *args, **kwargs):
        beta, gamma = self.x_to_beta_gamma(x)
        return self.loss(beta, gamma, **kwargs)

    def joint_gradient(self, x, *args, **kwargs):
        beta, gamma = self.x_to_beta_gamma(x)
        gradient = np.zeros(len(x))
        gradient[:self.problem.num_fixed_effects] = self.gradient_beta(beta, gamma)
        gradient[self.problem.num_fixed_effects:self.problem.num_fixed_effects + self.problem.num_random_effects] = \
            self.gradient_gamma(beta, gamma)
        return gradient

    def value_function(self, w, **kwargs):
        return self.joint_loss(w, **kwargs)

    def gradient_value_function(self, w):
        return self.joint_gradient(w)

    def optimal_beta(self, gamma: np.ndarray, _dont_solve_wrt_beta=False, **kwargs):
        """
        Returns beta (vector of estimations of fixed effects) which minimizes loss function for a fixed gamma.

        The algorithm for computing optimal beta is::

            kernel = ∑X_i^TΩ_iX_i

            tail = ∑X_i^TΩ_iY_i

            β = (kernel)^{-1}*tail

        It's available almost exclusively in linear models. In general one should use gradient_beta and do iterative
        minimization instead.

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
            Set of optimal random effects estimations for given beta and gamma

        """

        random_effects = []
        self._recalculate_cholesky(gamma)
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            stds_inv_mat = np.diag(1 / stds)
            # If the variance of R.E. is 0 then the R.E. is 0, so we take it into account separately
            # to keep matrices invertible.
            mask = np.abs(gamma) > 1e-10
            z_masked = z[:, mask]
            gamma_masked = gamma[mask]
            u_nonzero = np.linalg.solve(np.diag(1 / gamma_masked) + z_masked.T.dot(stds_inv_mat).dot(z_masked),
                                        z_masked.T.dot(stds_inv_mat).dot(xi)
                                        )
            u = np.zeros(len(gamma))
            u[mask] = u_nonzero
            random_effects.append(u)
        return np.array(random_effects)

    def optimal_obs_std(self, beta, gamma, **kwargs):
        self._recalculate_cholesky(gamma)
        result = 0
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            r = y - x.dot(beta)
            result += sum(L_inv.dot(r) ** 2)
        return result / self.problem.num_obs

    def _jones2010n_eff(self):
        n_eff = 0
        for (x, y, z, stds), L in zip(self.problem, self.omega_cholesky):
            omega = L.dot(L.T)
            sigma = np.sqrt(np.diag(omega)).reshape((-1, 1))
            normalization = sigma.dot(sigma.T)
            c = omega / normalization
            n_eff += np.linalg.inv(c).sum()
        return n_eff

    def jones2010bic(self, beta, gamma, **kwargs):
        # From here
        # https://www.researchgate.net/publication/51536734_Bayesian_information_criterion_for_longitudinal_and_clustered_data
        self._recalculate_cholesky(gamma)
        p = sum(beta != 0)
        q = sum(gamma != 0)
        return self.value_function(self.beta_gamma_to_x(beta, gamma)) + (p + q) * np.log(self._jones2010n_eff())

    def muller2018ic(self, beta, gamma, **kwargs):
        self._recalculate_cholesky(gamma)
        N = self.problem.num_obs
        n_eff = self._jones2010n_eff()
        return 2 / N * self.value_function(self.beta_gamma_to_x(beta, gamma)) \
               + 1 / N * np.log(n_eff) * sum(beta != 0) \
               + 2 / N * sum(gamma != 0)

    def _hat_matrix(self, gamma):
        self._recalculate_cholesky(gamma)

        h_beta_kernel = 0
        h_beta_tail = []
        xs = []
        # Form the beta kernel and tail
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            Lx = L_inv.dot(x)
            h_beta_kernel += Lx.T.dot(Lx)
            h_beta_tail.append(Lx.T.dot(L_inv))
            xs.append(x)

        h_beta = np.concatenate([np.linalg.inv(h_beta_kernel).dot(tail) for tail in h_beta_tail], axis=1)
        xs = np.concatenate(xs, axis=0)
        h_beta = xs.dot(h_beta)

        # we treat R.E. with very small variance
        # as effectively no R.E. to improve the stability of matrix inversions
        mask = np.abs(gamma) > 1e-10

        random_effects_parts = []
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            # Form the h_gamma
            stds_inv_mat = np.diag(1 / stds)
            # If the variance of R.E. is 0 then the R.E. is 0, so we take it into account separately
            # to keep matrices invertible.
            z_masked = z[:, mask]
            gamma_masked = gamma[mask]
            h_gamma_kernel = np.diag(1 / gamma_masked) + z_masked.T.dot(stds_inv_mat).dot(z_masked)
            h_gamma_tail = z_masked.T.dot(stds_inv_mat)
            h_gamma_i = np.zeros((self.problem.num_random_effects, z.shape[0]))
            h_gamma_i[mask, :] = np.linalg.inv(h_gamma_kernel).dot(h_gamma_tail)
            random_effects_parts.append(z.dot(h_gamma_i))

        h_gamma = sp.linalg.block_diag(*random_effects_parts)
        h = h_beta + h_gamma - h_gamma.dot(h_beta)
        return h

    def _hodges2001ddf(self, gamma, **kwargs):
        # From here:
        # https://www.jstor.org/stable/2673485?seq=1

        h_matrix = self._hat_matrix(gamma)
        return np.trace(h_matrix)

    def vaida2005aic(self, beta, gamma, **kwargs):
        # From here
        # https://www.jstor.org/stable/2673485?seq=1
        rho = self._hodges2001ddf(gamma)
        n = self.problem.num_obs
        p = sum(beta != 0)
        q = sum(gamma != 0)
        alpha = 2 * n / (n - p - 2) * (rho - (rho - p) / (n - p))
        # The likelihood here is conditional in the original paper
        # i.e. L(beta, gamma, us), but I put marginalized likelihood instead.
        return 2 * self.value_function(self.beta_gamma_to_x(beta, gamma), **kwargs) + alpha * (p + q)

    def get_ic(self, ic, beta, gamma, **kwargs):
        if ic == "IC_vaida2005aic":
            return self.vaida2005aic(beta, gamma, **kwargs)
        elif ic == "IC_jones2010bic":
            return self.jones2010bic(beta, gamma, **kwargs)
        else:
            raise ValueError(f"Unknown information criterion: {ic}")

    def get_condition_numbers(self):
        singv_x = []
        singv_z = []
        for x, y, z, l in self.problem:
            U, Sx, V = np.linalg.svd(x.T.dot(x))
            U, Sz, V = np.linalg.svd(z.T.dot(z))
            singv_x += list(Sx)
            singv_z += list(Sz)

        return (np.infty if min(singv_x) == 0 else max(singv_x) / min(singv_x),
                np.infty if min(singv_z) == 0 else max(singv_z) / min(singv_z))


class LinearLMEOracleSR3(LinearLMEOracle):
    """
       Implements Regularized Linear Mixed-Effects Model functional for given problem::

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

    def __init__(self, problem: Optional[LinearLMEProblem], lb=0.1, lg=0.1, warm_start=True, **kwargs):
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
        """

        super().__init__(problem, **kwargs)
        self.lb = lb
        self.lg = lg
        self.warm_start = warm_start
        self.warm_start_ip = {}

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

        return (super().loss(beta, gamma, **kwargs)
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

        return super().gradient_gamma(beta, gamma, **kwargs) + self.lg * (gamma - tgamma)

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

        return super().hessian_gamma(beta, gamma, **kwargs) + self.lg * np.eye(self.problem.num_random_effects)

    def gradient_beta(self, beta: np.ndarray, gamma: np.ndarray, tbeta: np.ndarray = None, **kwargs) -> np.ndarray:
        return super().gradient_beta(beta, gamma, **kwargs) + self.lb * (beta - tbeta)

    def hessian_beta(self, beta: np.ndarray, gamma: np.ndarray, **kwargs):
        return super().hessian_beta(beta, gamma, **kwargs) + self.lb * np.eye(self.problem.num_fixed_effects)

    def joint_loss(self, x, w, *args, **kwargs):
        beta, gamma = self.x_to_beta_gamma(x)
        return self.loss(beta, gamma)

    def joint_gradient(self, x, tbeta=None, tgamma=None):
        beta, gamma = self.x_to_beta_gamma(x)
        gradient = np.zeros(len(x))
        gradient[:self.problem.num_fixed_effects] = self.gradient_beta(beta, gamma, tbeta=tbeta)
        gradient[self.problem.num_fixed_effects:self.problem.num_fixed_effects + self.problem.num_random_effects] = \
            self.gradient_gamma(beta, gamma, tgamma=tgamma)
        return gradient

    def value_function(self, w):
        tbeta, tgamma = self.x_to_beta_gamma(w)
        beta, gamma, tbeta, tgamma, log = self.find_optimal_parameters_ip(2 * tbeta, 2 * tgamma, tbeta=tbeta,
                                                                          tgamma=tgamma)
        return self.loss(beta, gamma, tbeta=tbeta, tgamma=tgamma)

    def gradient_value_function(self, w):
        tbeta, tgamma = self.x_to_beta_gamma(w)
        beta, gamma, tbeta, tgamma, log = self.find_optimal_parameters_ip(2 * tbeta, 2 * tgamma, tbeta=tbeta,
                                                                          tgamma=tgamma)
        x = self.beta_gamma_to_x(beta, gamma)
        lambdas = np.array([self.lb] * self.problem.num_fixed_effects + [self.lg] * self.problem.num_random_effects)
        return -lambdas * (x - w)

    def find_optimal_parameters(self, w, log_progress=False, regularizer=None, increase_lambdas=False,
                                line_search=False, prox_step_len=1.0, update_prox_every=1,
                                **kwargs):
        tbeta, tgamma = self.x_to_beta_gamma(w)
        beta, gamma, tbeta, tgamma, log = self.find_optimal_parameters_ip(beta=2 * tbeta,
                                                                          gamma=2 * tgamma,
                                                                          tbeta=tbeta,
                                                                          tgamma=tgamma,
                                                                          log_progress=log_progress,
                                                                          regularizer=regularizer,
                                                                          increase_lambdas=increase_lambdas,
                                                                          line_search=line_search,
                                                                          prox_step_len=prox_step_len,
                                                                          update_prox_every=update_prox_every,
                                                                          **kwargs)
        return self.beta_gamma_to_x(tbeta, tgamma)

    def find_optimal_parameters_ip(self, beta: np.ndarray, gamma: np.ndarray, tbeta=None, tgamma=None,
                                   log_progress=False, regularizer=None, increase_lambdas=False,
                                   line_search=False, prox_step_len=1.0, update_prox_every=1,
                                   **kwargs):
        n = len(gamma)
        I = np.eye(n)
        Zb = np.zeros((len(gamma), len(beta)))
        v = np.ones(n)

        if self.warm_start:
            beta = self.warm_start_ip.get("beta", beta)
            gamma = self.warm_start_ip.get("gamma", gamma)

        # The packing of variables is x = [v (dual for gamma), beta, gamma]
        # All Lagrange gradients (F) and hessians (dF) have the same order of blocks.
        x = np.concatenate([v, beta, gamma])
        mu = 0.1 * v.dot(gamma) / n
        step_len = 1
        iteration = 0
        if log_progress:
            self.logger = [gamma]
        losses_kkt = []
        F_coord = lambda v, b, g, mu: np.concatenate([
            v * g - mu,
            self.gradient_beta(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs),
            self.gradient_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs) - v
        ])
        F = lambda x, mu: F_coord(x[:n], x[n:-n], x[-n:], mu)

        prev_tbeta = np.infty
        prev_tgamma = np.infty
        prev_beta = np.infty
        prev_gamma = np.infty

        tbeta_tgamma_convergence = False

        while step_len != 0 \
                and iteration < self.n_iter_inner \
                and np.linalg.norm(F(x, mu)) > self.tol_inner \
                and (np.linalg.norm(tbeta - prev_tbeta) > self.tol_inner
                     or np.linalg.norm(tgamma - prev_tgamma) > self.tol_inner
                     or np.linalg.norm(beta - prev_beta) > self.tol_inner
                     or np.linalg.norm(gamma - prev_gamma) > self.tol_inner
                     or tbeta_tgamma_convergence):
            prev_beta = beta
            prev_gamma = gamma
            prev_tbeta = tbeta
            prev_tgamma = tgamma

            F_coord = lambda v, b, g, mu: np.concatenate([
                v * g - mu,
                self.gradient_beta(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs),
                self.gradient_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs) - v
            ])
            F = lambda x, mu: F_coord(x[:n], x[n:-n], x[-n:], mu)
            dF_coord = lambda v, b, g: np.block([
                [np.diag(g), Zb, np.diag(v)],
                [Zb.T, self.hessian_beta(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs),
                 self.hessian_beta_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs)],
                [-I, self.hessian_beta_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs).T,
                 self.hessian_gamma(b, g, tbeta=tbeta, tgamma=tgamma, take_only_positive_part=True, **kwargs)]
            ])
            dF = lambda x: dF_coord(x[:n], x[n:-n], x[-n:])
            F_current = F(x, mu)
            dF_current = dF(x)
            direction = np.linalg.solve(dF_current, -F_current)
            # Determining maximal step size (such that gamma >= 0 and v >= 0)
            ind_neg_dir = np.where(direction < 0.0)[0]
            ind_neg_dir = ind_neg_dir[(ind_neg_dir < n) | (ind_neg_dir >= (len(x) - n))]
            max_step_len = min(1, 1 if len(ind_neg_dir) == 0 else np.min(-x[ind_neg_dir] / direction[ind_neg_dir]))

            if line_search:
                res = sp.optimize.minimize(fun=lambda alpha: np.linalg.norm(F(x + alpha * direction, mu)) ** 2,
                                           x0=np.array([max_step_len]),
                                           method="TNC",
                                           jac=lambda alpha: 2 * F(x + alpha * direction, mu).dot(
                                               dF(x + alpha * direction).dot(direction)),
                                           bounds=[(0, max_step_len)])
                step_len = res.x
            else:
                step_len = 0.99 * max_step_len
            x = x + step_len * direction
            # x[x <= 1e-18] = 0  # killing effective zeros
            v = x[:n]
            beta = x[n:-n]
            gamma = x[-n:]

            iteration += 1

            # optimize other components
            if regularizer and update_prox_every > 0 and iteration % update_prox_every == 0:
                tx = regularizer.prox(self.beta_gamma_to_x(beta=beta, gamma=gamma), alpha=prox_step_len)
                tbeta, tgamma = self.x_to_beta_gamma(tx)

            # losses.append(np.linalg.norm(F(x, mu)))
            losses_kkt.append(np.linalg.norm(F(x, mu)))

            if log_progress:
                self.logger.append(x[n:])

            # adjust barrier relaxation
            mu = 0.1 * v.dot(gamma) / n

            if increase_lambdas:
                self.lb = 1.2 * (1 + self.lb)
                self.lg = 1.2 * (1 + self.lg)
                tbeta_tgamma_convergence = (np.linalg.norm(beta - tbeta) > self.tol_inner
                                            or np.linalg.norm(gamma - tgamma) > self.tol_inner)
                # optimize other components
        if regularizer and update_prox_every > 0 and iteration < update_prox_every:
            tx = regularizer.prox(self.beta_gamma_to_x(beta=beta, gamma=gamma), alpha=prox_step_len)
            tbeta, tgamma = self.x_to_beta_gamma(tx)

        if self.warm_start:
            self.warm_start_ip["beta"] = beta
            self.warm_start_ip["gamma"] = gamma
        return beta, gamma, tbeta, tgamma, losses_kkt


class LinearLMEOracleRegularized(LinearLMEOracle):
    """
    Implements Regularized Linear Mixed-Effects Model functional for given problem::

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

    def __init__(self, problem: LinearLMEProblem, lb=0.1, lg=0.1, nnz_tbeta=3, nnz_tgamma=3,
                 participation_in_selection=None, independent_beta_and_gamma=False, **kwargs):
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
        participation_in_selection : Tuple of Int, Optional, default = None
            Which features participate in selection. Defaults to None, which means all features participate in
            selection process
        independent_beta_and_gamma: bool, Optional, default = False
            True if we don't need to set gammas to 0 whenever their respective betas are zero
        """

        super().__init__(problem, **kwargs)
        self.lb = lb
        self.lg = lg
        self.k = nnz_tbeta
        self.j = nnz_tgamma
        self.participation_in_selection = participation_in_selection
        self.independent_beta_and_gamma = independent_beta_and_gamma

    def optimal_beta(self, gamma: np.ndarray, tbeta: np.ndarray = None, _dont_solve_wrt_beta=False, **kwargs):
        """
        Returns beta (vector of estimations of fixed effects) which minimizes loss function for a fixed gamma.

        The algorithm for computing optimal beta is::

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
        kernel, tail = super().optimal_beta(gamma, _dont_solve_wrt_beta=True, **kwargs)
        if _dont_solve_wrt_beta:
            return kernel, tail
        return np.linalg.solve(self.lb * np.eye(self.problem.num_fixed_effects) + kernel, self.lb * tbeta + tail)

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
                                                                            self.k - sum(
                                                                                ~self.participation_in_selection))
            return result
        else:
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

        return (super().loss(beta, gamma, **kwargs)
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

        return super().gradient_gamma(beta, gamma, **kwargs) + self.lg * (gamma - tgamma)

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

        return super().hessian_gamma(beta, gamma, **kwargs) + self.lg * np.eye(self.problem.num_random_effects)

    def gradient_beta(self, beta: np.ndarray, gamma: np.ndarray, tbeta: np.ndarray = None, **kwargs) -> np.ndarray:
        return super().gradient_beta(beta, gamma, **kwargs) + self.lb * (beta - tbeta)

    def hessian_beta(self, beta: np.ndarray, gamma: np.ndarray, **kwargs):
        return super().hessian_beta(beta, gamma, **kwargs) + self.lb * np.eye(self.problem.num_fixed_effects)

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
            return self._take_only_k_max(tgamma, self.j)

        idx = tbeta == 0
        idx_gamma = self.beta_to_gamma_map[idx]
        idx_gamma = (idx_gamma[idx_gamma >= 0]).astype(int)
        tgamma[idx_gamma] = 0
        if self.participation_in_selection is not None:
            participation_idx = self.beta_to_gamma_map[self.participation_in_selection]
            participation_idx = (participation_idx[participation_idx >= 0]).astype(int)
            # if tbeta = 0 then tgamma = 0 even if this coordinate does not participate in feature selection
            not_participation_idx = self.beta_to_gamma_map[~self.participation_in_selection & (tbeta != 0)]
            not_participation_idx = (not_participation_idx[not_participation_idx >= 0]).astype(int)
            tgamma[not_participation_idx] = gamma[not_participation_idx]
            tgamma[participation_idx] = self._take_only_k_max(tgamma[participation_idx],
                                                              self.j - sum(~self.participation_in_selection))
            return tgamma
        else:
            return self._take_only_k_max(tgamma, self.j)

    def find_optimal_parameters_ip(self, beta: np.ndarray, gamma: np.ndarray, tbeta=None, tgamma=None,
                                   log_progress=False, increase_lanbdas=False,
                                   **kwargs):
        n = len(gamma)
        I = np.eye(n)
        Zb = np.zeros((len(gamma), len(beta)))
        v = np.ones(n)
        # The packing of variables is x = [v (dual for gamma), beta, gamma]
        # All Lagrange gradients (F) and hessians (dF) have the same order of blocks.
        x = np.concatenate([v, beta, gamma])
        mu = 0.1 * v.dot(gamma) / n
        step_len = 1
        iteration = 0
        if log_progress:
            self.logger = [gamma]
        losses_kkt = []
        F_coord = lambda v, b, g, mu: np.concatenate([
            v * g - mu,
            self.gradient_beta(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs),
            self.gradient_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs) - v
        ])
        F = lambda x, mu: F_coord(x[:n], x[n:-n], x[-n:], mu)

        prev_tbeta = np.infty
        prev_tgamma = np.infty
        prev_beta = np.infty
        prev_gamma = np.infty

        tbeta_tgamma_convergence = False

        while step_len != 0 \
                and iteration < self.n_iter_inner \
                and np.linalg.norm(F(x, mu)) > self.tol_inner \
                and (np.linalg.norm(tbeta - prev_tbeta) > self.tol_inner
                     or np.linalg.norm(tgamma - prev_tgamma) > self.tol_inner
                     or np.linalg.norm(beta - prev_beta) > self.tol_inner
                     or np.linalg.norm(gamma - prev_gamma) > self.tol_inner
                     or tbeta_tgamma_convergence):
            prev_beta = beta
            prev_gamma = gamma
            prev_tbeta = tbeta
            prev_tgamma = tgamma

            F_coord = lambda v, b, g, mu: np.concatenate([
                v * g - mu,
                self.gradient_beta(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs),
                self.gradient_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs) - v
            ])
            F = lambda x, mu: F_coord(x[:n], x[n:-n], x[-n:], mu)
            dF_coord = lambda v, b, g: np.block([
                [np.diag(g), Zb, np.diag(v)],
                [Zb.T, self.hessian_beta(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs),
                 self.hessian_beta_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs)],
                [-I, self.hessian_beta_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs).T,
                 self.hessian_gamma(b, g, tbeta=tbeta, tgamma=tgamma, take_only_positive_part=True, **kwargs)]
            ])
            dF = lambda x: dF_coord(x[:n], x[n:-n], x[-n:])
            F_current = F(x, mu)
            dF_current = dF(x)
            direction = np.linalg.solve(dF_current, -F_current)
            # Determining maximal step size (such that gamma >= 0 and v >= 0)
            ind_neg_dir = np.where(direction < 0.0)[0]
            ind_neg_dir = ind_neg_dir[(ind_neg_dir < n) | (ind_neg_dir >= (len(x) - n))]
            max_step_len = min(1, 1 if len(ind_neg_dir) == 0 else np.min(-x[ind_neg_dir] / direction[ind_neg_dir]))
            # res = sp.optimize.minimize(fun=lambda alpha: np.linalg.norm(F(x + alpha * direction, mu)) ** 2,
            #                            x0=np.array([max_step_len]),
            #                            method="TNC",
            #                            jac=lambda alpha: 2 * F(x + alpha * direction, mu).dot(
            #                                dF(x + alpha * direction).dot(direction)),
            #                            bounds=[(0, max_step_len)])
            # step_len = res.x
            step_len = 0.99 * max_step_len
            x = x + step_len * direction
            # x[x <= 1e-18] = 0  # killing effective zeros
            v = x[:n]
            beta = x[n:-n]
            gamma = x[-n:]
            # optimize other components
            tbeta = self.optimal_tbeta(beta=beta, gamma=gamma)
            tgamma = self.optimal_tgamma(tbeta, gamma, beta=beta)
            # adjust barrier relaxation
            mu = 0.1 * v.dot(gamma) / n
            if increase_lanbdas:
                self.lb = 1.2 * (1 + self.lb)
                self.lg = 1.2 * (1 + self.lg)
                tbeta_tgamma_convergence = (np.linalg.norm(beta - tbeta) > self.tol_inner
                                            or np.linalg.norm(gamma - tgamma) > self.tol_inner)

            iteration += 1
            # losses.append(np.linalg.norm(F(x, mu)))
            losses_kkt.append(np.linalg.norm(F(x, mu)))

            if log_progress:
                self.logger.append(x[n:])
        return beta, gamma, tbeta, tgamma, losses_kkt

    def find_optimal_parameters_pgd(self, beta: np.ndarray, gamma: np.ndarray, tbeta=None, tgamma=None,
                                    log_progress=False,
                                    **kwargs):
        step_len = 1
        iteration = 0

        prev_tbeta = np.infty
        prev_tgamma = np.infty
        prev_beta = np.infty
        prev_gamma = np.infty

        direction = np.infty
        proj_direction = np.infty

        if log_progress:
            self.logger = [gamma]
        losses = []

        while step_len > 0 \
                and iteration < self.n_iter_inner \
                and np.linalg.norm(proj_direction) > self.tol_inner \
                and (np.linalg.norm(tbeta - prev_tbeta) > self.tol_inner
                     or np.linalg.norm(tgamma - prev_tgamma) > self.tol_inner
                     or np.linalg.norm(beta - prev_beta) > self.tol_inner
                     or np.linalg.norm(gamma - prev_gamma) > self.tol_inner):

            prev_beta = beta
            prev_gamma = gamma
            prev_tbeta = tbeta
            prev_tgamma = tgamma

            beta = self.optimal_beta(gamma, tbeta, beta=beta)
            direction = -self.gradient_gamma(beta, gamma, tbeta=tbeta, tgamma=tgamma, **kwargs)
            # projecting the direction onto the constraints (positive box for gamma)
            proj_direction = direction.copy()
            proj_direction[(direction < 0) & (gamma == 0.0)] = 0

            res = sp.optimize.minimize(
                fun=lambda a: self.loss(beta, np.clip(gamma + a * direction, 0, None), tbeta=tbeta, tgamma=tgamma,
                                        **kwargs),
                x0=np.array([0]),
                # TODO: figure out how to get gradients back (projecting directions?)
                # method="TNC",
                # jac=lambda a: direction.dot(self.gradient_gamma(beta, np.clip(gamma + a * direction, 0, None), tbeta=tbeta, tgamma=tgamma, **kwargs)),
                bounds=[(0, 1)]
            )
            step_len = res.x
            gamma = np.clip(gamma + step_len * direction, 0, None)
            gamma[gamma <= 1e-18] = 0  # killing effective zeros
            # optimize other components
            tbeta = self.optimal_tbeta(beta=beta, gamma=gamma)
            tgamma = self.optimal_tgamma(tbeta, gamma, beta=beta)
            losses.append(self.loss(beta, gamma, tbeta, tgamma, **kwargs))
            iteration += 1
            if log_progress:
                self.logger.append(gamma)

        return beta, gamma, tbeta, tgamma, losses


class LinearLMEOracleW(LinearLMEOracleRegularized):

    def __init__(self, problem: LinearLMEProblem, lb=0.1, lg=0.1, nnz_tbeta=3, nnz_tgamma=3, **kwargs):
        super().__init__(problem, lb, lg, nnz_tbeta, nnz_tgamma, **kwargs)
        self.beta = None
        self.drop_penalties_beta = None
        self.drop_penalties_gamma = None

    def _recalculate_drop_matrices(self, beta, gamma):
        if np.all(self.gamma == gamma) and np.all(self.beta == beta):
            return None
        self._recalculate_cholesky(gamma)

        self.drop_penalties_beta = np.zeros(self.problem.num_fixed_effects)
        self.drop_penalties_gamma = np.zeros(self.problem.num_random_effects)
        for j, ((x, y, z, l), L_inv) in enumerate(zip(self.problem, self.omega_cholesky_inv)):
            # Calculate drop price for gammas individually
            xi = y - x.dot(beta)
            Lxi = L_inv.dot(xi)
            Lx = L_inv.dot(x)
            Lz = L_inv.dot(z)
            h1 = np.sum(Lz ** 2, axis=0)
            g1 = Lz.T.dot(Lxi) ** 2
            self.drop_penalties_gamma += -gamma * g1 / (1 - gamma * h1) + np.log(1 + gamma * h1 / (1 - gamma * h1))
            # Calculate drop price for betas only
            self.drop_penalties_beta += -2 * beta * Lx.T.dot(Lxi)
            self.drop_penalties_beta += -beta ** 2 * np.sum(Lx ** 2, axis=0)
            # Calculate drop price for gammas given dropped betas
            idx_beta = []
            idx_gamma = []
            for i, k in enumerate(self.beta_to_gamma_map):
                if k >= 0:
                    idx_beta.append(i)
                    idx_gamma.append(k)
            idx_beta = np.array(idx_beta)
            idx_gamma = np.array(idx_gamma).astype(int)
            x_s = x[:, idx_beta]
            beta_s = beta[idx_beta]
            gamma_s = gamma[idx_gamma]
            z_s = z[:, idx_gamma]
            Lz_s = L_inv.dot(z_s)
            h1_s = np.sum(Lz_s ** 2, axis=0)
            g2_s = np.sum((Lxi.reshape((len(Lxi), 1)) + L_inv.dot(x_s * beta_s)) * Lz_s, axis=0)
            self.drop_penalties_beta[idx_beta] += (-gamma_s * (g2_s ** 2) / (1 - gamma_s * h1_s)
                                                   + np.log(1 + gamma_s * h1_s / (1 - gamma_s * h1_s)))

        # we invert the sign and take into account the 1/2 multiplier for the loss function
        self.drop_penalties_beta /= -2
        self.drop_penalties_gamma /= -2
        self.beta = beta
        return None

    def loss(self, beta: np.ndarray, gamma: np.ndarray, tbeta: np.ndarray = None, tgamma: np.ndarray = None, **kwargs):
        if self.drop_penalties_beta is None or self.drop_penalties_gamma is None:
            self._recalculate_drop_matrices(beta, gamma)
        return (super(LinearLMEOracleRegularized, self).loss(beta, gamma, **kwargs)
                + self.lb / 2 * sum(self.drop_penalties_beta * (beta - tbeta) ** 2)
                + self.lg / 2 * sum(self.drop_penalties_gamma * (gamma - tgamma) ** 2))

    def optimal_beta(self, gamma: np.ndarray, tbeta: np.ndarray = None, beta: np.ndarray = None, **kwargs):
        if self.drop_penalties_beta is None:
            if beta is not None:
                self._recalculate_drop_matrices(beta, gamma)
            else:
                raise ValueError("Drop penalties for beta are not initialized")

        kernel, tail = super(LinearLMEOracleRegularized, self).optimal_beta(gamma, _dont_solve_wrt_beta=True, **kwargs)
        return np.linalg.solve(self.lb * np.diag(self.drop_penalties_beta) + kernel,
                               self.lb * self.drop_penalties_beta * tbeta + tail)

    def gradient_gamma(self, beta: np.ndarray, gamma: np.ndarray, tgamma: np.ndarray = None, **kwargs) -> np.ndarray:
        if self.drop_penalties_gamma is None:
            self._recalculate_drop_matrices(beta, gamma)
        return (super(LinearLMEOracleRegularized, self).gradient_gamma(beta, gamma, tgamma=tgamma, **kwargs)
                + self.lg * self.drop_penalties_gamma * (gamma - tgamma))

    def hessian_gamma(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        if self.drop_penalties_gamma is None:
            self._recalculate_drop_matrices(beta, gamma)
        return super(LinearLMEOracleRegularized, self).hessian_gamma(beta, gamma, **kwargs) + self.lg * np.diag(
            self.drop_penalties_gamma)

    def optimal_tbeta(self, beta: np.ndarray, gamma: np.ndarray = None, **kwargs):
        self._recalculate_drop_matrices(beta, gamma)
        tbeta = np.zeros(len(beta))
        idx_k_max = np.abs(self.drop_penalties_beta * beta).argsort()[-self.k:]
        tbeta[idx_k_max] = beta[idx_k_max]
        return tbeta

    def optimal_tgamma(self, tbeta, gamma, beta: np.ndarray = None, **kwargs):
        self._recalculate_drop_matrices(beta, gamma)
        tgamma = np.zeros(len(gamma))
        idx = tbeta != 0
        idx_gamma = self.beta_to_gamma_map[idx]
        idx_gamma = (idx_gamma[idx_gamma >= 0]).astype(int)
        tgamma[idx_gamma] = gamma[idx_gamma]
        idx_k_max = np.abs(self.drop_penalties_gamma * tgamma).argsort()[-self.j:]
        tgamma2 = np.zeros(len(gamma))
        tgamma2[idx_k_max] = tgamma[idx_k_max]
        return tgamma2

    def jones2010bic(self, beta, gamma, tbeta=None, tgamma=None, **kwargs):
        self._recalculate_cholesky(gamma)
        return self.loss(beta, gamma, tbeta, tgamma) + (2 * len(beta) + 2 * len(gamma)) * np.log(self._jones2010n_eff())
