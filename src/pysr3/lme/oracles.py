# This code implements linear mixed-effects oracle as a subroutine for skmixed subroutines.
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
from typing import Optional

import numpy as np
import scipy as sp
from scipy import stats
from scipy.linalg.lapack import get_lapack_funcs
from scipy.optimize import minimize

from pysr3.lme.priors import NonInformativePriorLME
from pysr3.lme.problems import LMEProblem, FIXED, RANDOM, FIXED_RANDOM


class LinearLMEOracle:
    """
    Implements Linear Mixed-Effects Model functional for given problem.

    The model is::

        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        u_i ~ 𝒩(0, diag(𝛄)),

        𝜺_i ~ 𝒩(0, Λ)

    The problem should be provided as LMEProblem.

    """

    def __init__(self, problem: Optional[LMEProblem],
                 n_iter_inner=200,
                 tol_inner=1e-6,
                 warm_start_duals=False,
                 prior=None):
        """
        Creates an oracle on top of the given problem

        Parameters
        ----------
        problem : LMEProblem
            set of data and answers. See docs for LinearLMEProblem class for more details.
        n_iter_inner : int
            maximal number of iterations for the oracle's numerical subroutines, if any
        tol_inner : float
            tolerance for stopping criteria of the oracle's numerical subroutines, if any
        warm_start_duals: bool
            if to warm-start when numerical subroutines are called sequentially
        prior:
            additional prior for the oracle, see skmixed.priors for more info.
        """
        self.problem = problem
        self.prior = prior if prior else NonInformativePriorLME()
        self.beta_to_gamma_map = None
        self.omega_cholesky_inv = []
        self.omega_cholesky = []
        self.gamma = None
        self.logger = None
        self.n_iter_inner = n_iter_inner
        self.tol_inner = tol_inner
        if warm_start_duals:
            self.v = None
        if problem:
            self.instantiate(problem)

    def instantiate(self, problem):
        """
        Attach the problem to the oracle

        Parameters
        ----------
        problem: LMEProblem
            problem to attach

        Returns
        -------
        None

        """
        self.problem = problem
        beta_to_gamma_map = np.zeros(self.problem.num_fixed_features)
        beta_counter = 0
        gamma_counter = 0
        for label in ([
            self.problem.intercept_label] if self.problem.intercept_label else []) + self.problem.column_labels:
            if label == FIXED:
                beta_to_gamma_map[beta_counter] = -1
                beta_counter += 1
            elif label == RANDOM:
                gamma_counter += 1
            elif label == FIXED_RANDOM:
                beta_to_gamma_map[beta_counter] = gamma_counter
                beta_counter += 1
                gamma_counter += 1
            else:
                continue
        self.beta_to_gamma_map = beta_to_gamma_map
        self.prior.instantiate(problem)

    def forget(self):
        """
        Detaches the problem from the oracle

        Returns
        -------
        None
        """
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

        if (self.gamma != gamma).any() or len(gamma) == 0:
            self.omega_cholesky = []
            self.omega_cholesky_inv = []
            gamma_mat = np.diag(gamma)
            invert_upper_triangular: Callable[[np.ndarray], np.ndarray] = get_lapack_funcs("trtri")
            for x, y, z, stds in self.problem:
                omega = np.diag(stds)
                if len(gamma) > 0:
                    omega += z.dot(gamma_mat).dot(z.T)
                el = np.linalg.cholesky(omega)
                el_inv = invert_upper_triangular(el.T)[0].T
                self.omega_cholesky.append(el)
                self.omega_cholesky_inv.append(el_inv)
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
        """
        Evaluates a de-marginalized loss of the model

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
        the value of the de-marginalized loss
        """
        result = 0
        self._recalculate_cholesky(gamma)
        us = self.optimal_random_effects(beta, gamma, **kwargs)
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
        for (x, y, z, stds), el_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            el_z = el_inv.dot(z)
            grad_gamma += 1 / 2 * np.sum(el_z ** 2, axis=0) - 1 / 2 * el_z.T.dot(el_inv.dot(xi)) ** 2
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
        take_only_positive_part: bool
            Whether to return only the positive-definite part of Hessian
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
            hessian: np.ndarray, shape = [k, k]
                Hessian of the loss function with respect to gamma ∇²_𝛄[ℒ](β, 𝛄).
        """
        self._recalculate_cholesky(gamma)
        num_random_effects = self.problem.num_random_features
        hessian = np.zeros(shape=(num_random_effects, num_random_effects))
        for (x, y, z, stds), el_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            el_z = el_inv.dot(z)
            el_xi = el_inv.dot(xi).reshape((len(xi), 1))
            hessian += ((0 if take_only_positive_part else -el_z.T.dot(el_z)) + 2 * (
                el_z.T.dot(el_xi).dot(el_xi.T).dot(el_z))) * (
                           el_z.T.dot(el_z))
        return 1 / 2 * hessian + self.prior.hessian_gamma(beta, gamma)

    def optimal_gamma_pgd(self, beta: np.ndarray, gamma: np.ndarray, log_progress=False, **kwargs):
        """
        Evaluates optimal gamma given beta with a Proximal Projected Gradient Descent

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects, initial point.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters
        log_progress : bool
            Whether to log the loss function during optimization (for debugging only)

        Returns
        -------
        optimal gamma
        """
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
        """
        Evaluates optimal gamma given beta with an Interior Point Method

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects, initial point.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters
        log_progress : bool
            Whether to log the loss function during optimization (for debugging only)

        Returns
        -------
        optimal gamma
        """
        n = len(gamma)
        eye = np.eye(n)

        # v = x[:n], g = x[n:]

        def F_coord(v, g, mu):
            return np.concatenate([
                v * g - mu,
                self.gradient_gamma(beta, g, **kwargs) - v
            ])

        def F(x, mu):
            return F_coord(x[:n], x[n:], mu)

        def dF_coord(v, g):
            return np.block([
                [np.diag(g), np.diag(v)],
                [-eye, self.hessian_gamma(beta, g, take_only_positive_part=True, **kwargs)]
            ])

        def dF(x):
            return dF_coord(x[:n], x[n:])

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
        """
        Evaluates optimal gamma given beta

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects, initial point.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters
        method : str
            which numerical subroutine to use: "pgd" or "ip"
        Returns
        -------
        optimal gamma
        """
        if method == "pgd":
            return self.optimal_gamma_pgd(beta, gamma, **kwargs)
        elif method == "ip":
            return self.optimal_gamma_ip(beta, gamma, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def gradient_beta(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates the gradient of the loss-function with respect to beta

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects, initial point.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        the gradient of the loss function with respect to beta
        """
        self._recalculate_cholesky(gamma)
        gradient = np.zeros(self.problem.num_fixed_features)
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            gradient += - (L_inv.dot(x)).T.dot(L_inv.dot(xi))
        return gradient + self.prior.gradient_beta(beta=beta, gamma=gamma)

    def hessian_beta(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates Hessian of the loss-function with respect to beta

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects, initial point.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        Hessian of the loss function with respect to beta
        """
        self._recalculate_cholesky(gamma)
        hessian = np.zeros((self.problem.num_fixed_features, self.problem.num_fixed_features))
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            Lx = L_inv.dot(x)
            hessian += Lx.T.dot(Lx)
        return hessian + self.prior.hessian_beta(beta=beta, gamma=gamma)

    def flip_probabilities_beta(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Estimates the probabilities of coordinates in beta to flip their signs
        under the normal posterior.

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects, initial point.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        probabilities each coordinates of beta, as ndarray.
        """
        hessian = self.hessian_beta(beta=beta, gamma=gamma)
        cov_matrix = np.linalg.inv(hessian)
        probabilities = []
        for mean, var, sign in zip(beta, np.diag(cov_matrix), np.sign(beta)):
            if sign > 0:
                probabilities.append(stats.norm.cdf(0, loc=mean, scale=np.sqrt(var)))
            elif sign < 0:
                probabilities.append(1 - stats.norm.cdf(0, loc=mean, scale=np.sqrt(var)))
            else:
                probabilities.append(0.5)
        return np.array(probabilities)

    def hessian_beta_gamma(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates mixed Hessian of the loss-function with respect to (beta, gamma)

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects, initial point.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        Hessian of the loss function with respect to (beta, gamma)
        """
        self._recalculate_cholesky(gamma)
        hessian = np.zeros((self.problem.num_random_features, self.problem.num_fixed_features))
        for (x, y, z, stds), el_inv in zip(self.problem, self.omega_cholesky_inv):
            xi = y - x.dot(beta)
            el_x = el_inv.dot(x)
            el_z = el_inv.dot(z)
            el_xi = el_inv.dot(xi)
            hessian += np.diag(el_z.T.dot(el_xi)).dot(el_z.T.dot(el_x))
        return hessian.T + self.prior.hessian_beta_gamma(beta=beta, gamma=gamma)

    def x_to_beta_gamma(self, x):
        """
        Takes x = [beta, gamma], splits it, and returns beta and gamma separately

        Parameters
        ----------
        x: ndarray, (p+q)
            vector of parameters [beta, gamma]

        Returns
        -------
        Tuple of numpy arrays: beta and gamma
        """
        beta = x[:self.problem.num_fixed_features]
        gamma = x[self.problem.num_fixed_features:self.problem.num_fixed_features + self.problem.num_random_features]
        return beta, gamma

    @staticmethod
    def beta_gamma_to_x(beta, gamma):
        """
        Merges beta and gamma into one array x = [beta, gamma]

        Parameters
        ----------
        beta : np.ndarray, shape = [n]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [k]
            Vector of estimates of random effects, initial point.

        Returns
        -------
        x -- the ndarray of [beta, gamma]
        """
        return np.concatenate([beta, gamma])

    def joint_loss(self, x, *args, **kwargs):
        """
        Takes x = [beta, gamma] and evaluates the loss with this beta and gamma.

        Parameters
        ----------
        x: ndarray, (p+q)
            vector of parameters [beta, gamma]
        args:
            for debugging and passing parameters
        kwargs:
            for debugging and passing parameters

        Returns
        -------
        Loss at x = [beta, gamma]
        """
        beta, gamma = self.x_to_beta_gamma(x)
        return self.loss(beta, gamma, **kwargs)

    def joint_gradient(self, x, *args, **kwargs):
        """
        Takes x = [beta, gamma] and evaluates the gradient of the loss with this beta and gamma.

        Parameters
        ----------
        x: ndarray, (p+q)
            vector of parameters [beta, gamma]
        args:
            for debugging and passing parameters
        kwargs:
            for debugging and passing parameters

        Returns
        -------
        Gradient of the loss at x = [beta, gamma]
        """
        beta, gamma = self.x_to_beta_gamma(x)
        gradient = np.zeros(len(x))
        gradient[:self.problem.num_fixed_features] = self.gradient_beta(beta, gamma)
        gradient[self.problem.num_fixed_features:self.problem.num_fixed_features + self.problem.num_random_features] = \
            self.gradient_gamma(beta, gamma)
        return gradient

    def value_function(self, w, **kwargs):
        """
        Evaluates value function of the problem.
        If no relaxation is applied then the value function is the loss function.

        Parameters
        ----------
        w: ndarray, (p+q)
            vector of parameters [beta, gamma]
        kwargs:
            for debugging and passing parameters

        Returns
        -------
        Value of the value function
        """
        return self.joint_loss(w, **kwargs)

    def gradient_value_function(self, w):
        """
        Returns the gradient of the value function.

        Parameters
        ----------
        w: ndarray, (p+q)
            vector of parameters [beta, gamma]

        Returns
        -------
        Gradient of the value function
        """
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
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        _dont_solve_wrt_beta : bool, Optional
            If true, then it does not perform the outer matrix inversion and returns the (kernel, tail) instead.
            It's left here for the purposes of use in child classes where both the kernel and the tail should be
            adjusted to account for regularization.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        beta: np.ndarray, shape = [p]
            Vector of optimal estimates of the fixed effects for given gamma.
        """
        self._recalculate_cholesky(gamma)
        kernel = 0
        tail = 0
        for (x, y, z, stds), el_inv in zip(self.problem, self.omega_cholesky_inv):
            el_x = el_inv.dot(x)
            kernel += el_x.T.dot(el_x)
            tail += el_x.T.dot(el_inv.dot(y))
        if _dont_solve_wrt_beta:
            return kernel, tail
        else:
            return np.linalg.solve(kernel, tail)

    def optimal_random_effects(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns set of optimal random effects estimations for given beta and gamma.

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        u : np.ndarray, shape = [m, q]
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
        """
        Evaluate the optimal (in the maximal likelihood sense) variances of the observation errors.
        It assumes that all errors have sigma*I covariance matrices, where sigma is a scalar.

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        value of sigma -- amplitude of the noise.

        """
        self._recalculate_cholesky(gamma)
        result = 0
        for (x, y, z, stds), L_inv in zip(self.problem, self.omega_cholesky_inv):
            r = y - x.dot(beta)
            result += sum(L_inv.dot(r) ** 2)
        return result / self.problem.num_obs

    def _jones2010n_eff(self):
        """
        The "effective number of objects" from (Jones, 2010).
        https://www.researchgate.net/publication/51536734_Bayesian_information_criterion_for_longitudinal_and_clustered_data
        It can be less than the total number of objects because the objects are correlated within groups.
        It can not be, however, smaller than a number of groups.

        Returns
        -------
        effective number of objects in the dataset

        """
        n_eff = 0
        for L in self.omega_cholesky:
            omega = L.dot(L.T)
            sigma = np.sqrt(np.diag(omega)).reshape((-1, 1))
            normalization = sigma.dot(sigma.T)
            c = omega / normalization
            n_eff += np.linalg.inv(c).sum()
        return n_eff

    def jones2010bic(self, beta, gamma, tolerance=0., **kwargs):
        """
        Implements Bayes Information Criterion (BIC) from (Jones, 2010)
        # https://www.researchgate.net/publication/51536734_Bayesian_information_criterion_for_longitudinal_and_clustered_data

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        tolerance : float, positive
            Threshold for absolute values of beta and gamma being considered zero.
            Should account for the finite tolerance of the numerical solver.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        Value of Jones's BIC
        """
        self._recalculate_cholesky(gamma)
        p = sum(np.abs(beta) >= tolerance)
        q = sum(np.abs(gamma) >= tolerance)
        return 2 * self.loss(beta, gamma, **kwargs) + (p + q) * np.log(
            self._jones2010n_eff())

    def muller_hui_2016ic(self, beta, gamma, tolerance=0., **kwargs):
        """
        Implements Information Criterion (IC) from (Muller, 2016)
        https://www.tandfonline.com/doi/full/10.1080/01621459.2016.1215989
        page 1326, equation 3

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        tolerance : float, positive
            Threshold for absolute values of beta and gamma being considered zero.
            Should account for the finite tolerance of the numerical solver.

        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        Value of Mueller's IC
        """
        self._recalculate_cholesky(gamma)
        N = self.problem.num_obs
        # n_eff = self._jones2010n_eff()
        m = self.problem.num_groups
        return 2 / N * self.loss(beta, gamma, **kwargs) \
               + 1 / N * np.log(m) * sum(np.abs(beta) >= tolerance) \
               + 2 / N * sum(np.abs(gamma) >= tolerance)

    def _hat_matrix(self, gamma):
        """
        Implements the "Hat Matrix" from https://www.jstor.org/stable/2673485

        Parameters
        ----------
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.

        Returns
        -------
        Hat matrix
        """
        self._recalculate_cholesky(gamma)

        h_beta_kernel = 0
        h_beta_tail = []
        xs = []
        # Form the beta kernel and tail
        for (x, y, z, stds), el_inv in zip(self.problem, self.omega_cholesky_inv):
            el_x = el_inv.dot(x)
            h_beta_kernel += el_x.T.dot(el_x)
            h_beta_tail.append(el_x.T.dot(el_inv))
            xs.append(x)

        h_beta = np.concatenate([np.linalg.inv(h_beta_kernel).dot(tail) for tail in h_beta_tail], axis=1)
        xs = np.concatenate(xs, axis=0)
        h_beta = xs.dot(h_beta)

        # we treat R.E. with very small variance
        # as effectively no R.E. to improve the stability of matrix inversions
        mask = np.abs(gamma) > 1e-10

        random_effects_parts = []
        for (x, y, z, stds), el_inv in zip(self.problem, self.omega_cholesky_inv):
            # Form the h_gamma
            stds_inv_mat = np.diag(1 / stds)
            # If the variance of R.E. is 0 then the R.E. is 0, so we take it into account separately
            # to keep matrices invertible.
            z_masked = z[:, mask]
            gamma_masked = gamma[mask]
            h_gamma_kernel = np.diag(1 / gamma_masked) + z_masked.T.dot(stds_inv_mat).dot(z_masked)
            h_gamma_tail = z_masked.T.dot(stds_inv_mat)
            h_gamma_i = np.zeros((self.problem.num_random_features, z.shape[0]))
            h_gamma_i[mask, :] = np.linalg.inv(h_gamma_kernel).dot(h_gamma_tail)
            random_effects_parts.append(z.dot(h_gamma_i))

        h_gamma = sp.linalg.block_diag(*random_effects_parts)
        h = h_beta + h_gamma - h_gamma.dot(h_beta)
        return h

    def _hodges2001ddf(self, gamma, **kwargs):
        """
        Estimates the effective number of degrees of freedom of the model:
        https://www.jstor.org/stable/2673485

        Parameters
        ----------
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        "effective number" of degrees of freedom. Can be non-integer.
        """

        h_matrix = self._hat_matrix(gamma)
        return np.trace(h_matrix)

    def vaida2005aic(self, beta, gamma, tolerance=0., marginalized=False, **kwargs):
        """
        Calculates Akaike Information Criterion (AIC) from https://www.jstor.org/stable/2673485

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        tolerance : float, positive
            Threshold for absolute values of beta and gamma being considered zero.
            Should account for the finite tolerance of the numerical solver.

        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        Value for Vaida AIC
        """
        n = self.problem.num_obs
        p = sum(np.abs(beta) >= tolerance)
        q = sum(np.abs(gamma) >= tolerance)
        if marginalized:
            # mAIC version
            # See also p 141 eq 10 here
            # https://projecteuclid.org/journals/statistical-science/volume-28/issue-2/Model-Selection-in-Linear-Mixed-Models/10.1214/12-STS410.short
            if (n - p - q - 1) > 0:
                alpha = n / (n - p - q - 1)
            else:
                alpha = 1
            return 2 * self.loss(beta, gamma, **kwargs) + 2 * alpha * (p + q)
        else:
            # cAIC version
            # See also p 143 eq 17 here
            # https://projecteuclid.org/journals/statistical-science/volume-28/issue-2/Model-Selection-in-Linear-Mixed-Models/10.1214/12-STS410.short
            rho = self._hodges2001ddf(gamma)
            alpha = 2 * n / (n - p - 2) * (rho - (rho - p) / (n - p))
            return 2 * self.demarginalized_loss(beta, gamma, **kwargs) + alpha * (p + q)

    def get_ic(self, ic, beta, gamma, **kwargs):
        """
        Wrapper for information criteria

        Parameters
        ----------
        ic: str
            Can be "IC_jones2010bic", "IC_vaida2005aic"
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters


        Returns
        -------
        The value of the information criterion requested
        """
        if ic == "IC_vaida2005aic":
            return self.vaida2005aic(beta, gamma, **kwargs)
        elif ic == "IC_jones2010bic":
            return self.jones2010bic(beta, gamma, **kwargs)
        else:
            raise ValueError(f"Unknown information criterion: {ic}")

    def get_condition_numbers(self):
        """
        Returns the condition numbers of the data matrices X and Z

        Returns
        -------
        Tuple of the condition numbers of the data matrices X and Z
        """
        singular_values_x = []
        singular_values_z = []
        for x, y, z, l in self.problem:
            u, sx, v = np.linalg.svd(x.T.dot(x))
            u, sz, v = np.linalg.svd(z.T.dot(z))
            singular_values_x += list(sx)
            singular_values_z += list(sz)

        return (np.infty if min(singular_values_x) == 0 else max(singular_values_x) / min(singular_values_x),
                np.infty if min(singular_values_z) == 0 else max(singular_values_z) / min(singular_values_z))


class LinearLMEOracleSR3(LinearLMEOracle):
    """
       Implements Sparse Relaxed Regularized Regression (SR3) for Linear Mixed-Effects Model.
       The problem should be provided as LMEProblem.

       """

    def __init__(self, problem: Optional[LMEProblem], lb=0.1, lg=0.1, warm_start=True, **kwargs):
        """
        Creates an oracle on top of the given problem. The problem should be in the form of LinearLMEProblem.

        Parameters
        ----------
        problem: LMEProblem
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
            Vector of relaxed estimates of fixed effects.
        tgamma : np.ndarray, shape = [k]
            Vector of relaxed estimates of random effects.
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
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        tgamma : np.ndarray, shape = [q]
            Vector of relaxed estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
            grad_gamma: np.ndarray, shape = [q]
                The gradient of the loss function with respect to gamma: grad_gamma = ∇_𝛄[ℒ](β, 𝛄) + lg*(𝛄 - t𝛄)
        """

        return super().gradient_gamma(beta, gamma, **kwargs) + self.lg * (gamma - tgamma)

    def hessian_gamma(self, beta: np.ndarray, gamma: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns the Hessian of the loss function with respect to gamma.

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
            hessian: np.ndarray, shape = [q, q]
                Hessian of the loss function with respect to gamma.
        """

        return super().hessian_gamma(beta, gamma, **kwargs) + self.lg * np.eye(self.problem.num_random_features)

    def gradient_beta(self, beta: np.ndarray, gamma: np.ndarray, tbeta: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Evaluates the gradient of the loss-function with respect to beta

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects, initial point.
        tbeta : np.ndarray, shape = [p]
            Vector of relaxed estimates of fixed effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        the gradient of the loss function with respect to beta
        """
        return super().gradient_beta(beta, gamma, **kwargs) + self.lb * (beta - tbeta)

    def hessian_beta(self, beta: np.ndarray, gamma: np.ndarray, **kwargs):
        """
        Returns the Hessian of the loss function with respect to gamma

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
            hessian: np.ndarray, shape = [q, q]
                Hessian of the loss function with respect to gamma
        """
        return super().hessian_beta(beta, gamma, **kwargs) + self.lb * np.eye(self.problem.num_fixed_features)

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
        return np.linalg.solve(self.lb * np.eye(self.problem.num_fixed_features) + kernel, self.lb * tbeta + tail)

    def joint_loss(self, x, w=None, *args, **kwargs):
        """
        Takes x = [beta, gamma], w = [tbeta, tgamma],
        and evaluates the loss with this beta and gamma.

        Parameters
        ----------
        x: ndarray, (p+q)
            vector of parameters [beta, gamma]
        w: ndarray, (p+q)
            vector of parameters [tbeta, tgamma]

        args:
            for debugging and passing parameters
        kwargs:
            for debugging and passing parameters

        Returns
        -------
        Loss at (x, w)
        """
        beta, gamma = self.x_to_beta_gamma(x)
        tbeta, tgamma = self.x_to_beta_gamma(w)
        return self.loss(beta, gamma, tbeta, tgamma, **kwargs)

    def joint_gradient(self, x, tbeta=None, tgamma=None):
        """
        Takes x = [beta, gamma] and evaluates the gradient of the loss with this beta and gamma.

        Parameters
        ----------
        x: ndarray, (p+q)
           vector of parameters [beta, gamma]
        tbeta : np.ndarray, shape = [p]
            Vector of relaxed estimates of fixed effects.
        tgamma : np.ndarray, shape = [q]
            Vector of relaxed estimates of random effects.
        Returns
        -------
        Gradient of the loss at x = [beta, gamma]
        """
        beta, gamma = self.x_to_beta_gamma(x)
        gradient = np.zeros(len(x))
        gradient[:self.problem.num_fixed_features] = self.gradient_beta(beta, gamma, tbeta=tbeta)
        gradient[self.problem.num_fixed_features:self.problem.num_fixed_features + self.problem.num_random_features] = \
            self.gradient_gamma(beta, gamma, tgamma=tgamma)
        return gradient

    def value_function(self, w, **kwargs):
        """
        Evaluates value function of the problem.

        Parameters
        ----------
        w: ndarray, (p+q)
            vector of parameters [beta, gamma]
        kwargs:
            for debugging and passing parameters

        Returns
        -------
        Value of the value function
        """
        tbeta, tgamma = self.x_to_beta_gamma(w)
        beta, gamma, tbeta, tgamma, log = self.find_optimal_parameters_ip(2 * tbeta, 2 * tgamma, tbeta=tbeta,
                                                                          tgamma=tgamma, **kwargs)
        return self.loss(beta, gamma, tbeta=tbeta, tgamma=tgamma, **kwargs)

    def gradient_value_function(self, w, logger=None, **kwargs):
        """
        Returns the gradient of the value function.

        Parameters
        ----------
        w: ndarray, (p+q)
            vector of parameters [beta, gamma]

        Returns
        -------
        Gradient of the value function
        """
        tbeta, tgamma = self.x_to_beta_gamma(w)

        beta, gamma, tbeta, tgamma, log = self.find_optimal_parameters_ip(2 * tbeta, 2 * tgamma, tbeta=tbeta,
                                                                          tgamma=tgamma)
        x = self.beta_gamma_to_x(beta, gamma)
        lambdas = np.array([self.lb] * self.problem.num_fixed_features + [self.lg] * self.problem.num_random_features)
        return -lambdas * (x - w)

    def find_optimal_parameters(self, w, log_progress=False, regularizer=None, increase_lambdas=False,
                                line_search=False, prox_step_len=1.0, update_prox_every=1,
                                **kwargs):
        """
        Wrapper around the routine that maximizes the likelihood of the oracle with respect to its
        non-relaxed parameters. This routine is designed for experiments doing all relaxations of (SR3 and IP)
        simultaneously (SR3Practical).

        Parameters
        ----------
        w: ndarray, (p+q)
            vector of parameters [beta, gamma]
        log_progress: bool
            whether to log progress of internal numerical routines
        regularizer: Regularizer
            an instance of Regularizer class
        increase_lambdas: bool
            whether to iteratively increase the parameters of SR3 relaxation
        line_search: bool
            whether to perform line-search inside IP method
        prox_step_len: float
            step-size for numerical subroutines
        update_prox_every: int
            how frequently algorithm updates sparse variables (tbeta, tgamma)
        kwargs:
            for passing extra arguments

        Returns
        -------
        Five objects: beta, gamma, tbeta, tgamma, logger
        """
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
                                   regularizer=None, increase_lambdas=False,
                                   line_search=False, prox_step_len=1.0, update_prox_every=1, logger=None,
                                   **kwargs):
        losses_kkt = []
        if len(tgamma) == 0:
            beta = self.optimal_beta(gamma=tgamma, tbeta=tbeta)
            gamma = tgamma
            return beta, gamma, tbeta, tgamma, losses_kkt

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

        def F_coord(v, b, g, mu):
            return np.concatenate([
                v * g - mu,
                self.gradient_beta(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs),
                self.gradient_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs) - v
            ])

        def F(x, mu):
            return F_coord(x[:n], x[n:-n], x[-n:], mu)

        def dF_coord(v, b, g):
            return np.block([
                [np.diag(g), Zb, np.diag(v)],
                [Zb.T, self.hessian_beta(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs),
                 self.hessian_beta_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs)],
                [-I, self.hessian_beta_gamma(b, g, tbeta=tbeta, tgamma=tgamma, **kwargs).T,
                 self.hessian_gamma(b, g, tbeta=tbeta, tgamma=tgamma, take_only_positive_part=True, **kwargs)]
            ])

        def dF(x):
            return dF_coord(x[:n], x[n:-n], x[-n:])

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

            # adjust barrier relaxation
            mu = 0.1 * v.dot(gamma) / n

            if increase_lambdas:
                self.lb = 1.2 * (1 + self.lb)
                self.lg = 1.2 * (1 + self.lg)
                tbeta_tgamma_convergence = (np.linalg.norm(beta - tbeta) > self.tol_inner
                                            or np.linalg.norm(gamma - tgamma) > self.tol_inner)
                # optimize other components (that was happening once at the end only, moved to each iteration
            if regularizer and update_prox_every > 0 and iteration < update_prox_every:
                tx = regularizer.prox(self.beta_gamma_to_x(beta=beta, gamma=gamma), alpha=prox_step_len)
                tbeta, tgamma = self.x_to_beta_gamma(tx)

            if logger and len(logger.keys) > 0:
                logger.log(locals())

        if self.warm_start:
            self.warm_start_ip["beta"] = beta
            self.warm_start_ip["gamma"] = gamma

        if logger:
            logger.add("iteration", iteration)

        return beta, gamma, tbeta, tgamma, losses_kkt

    def jones2010bic(self, beta, gamma, tolerance=0., **kwargs):
        """
        Implements Bayes Information Criterion (BIC) from (Jones, 2010)
        # https://www.researchgate.net/publication/51536734_Bayesian_information_criterion_for_longitudinal_and_clustered_data

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        tolerance : float, positive
            Threshold for absolute values of beta and gamma being considered zero.
            Should account for the finite tolerance of the numerical solver.
        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        Value of Jones's BIC
        """
        self._recalculate_cholesky(gamma)
        p = sum(np.abs(beta) >= tolerance)
        q = sum(np.abs(gamma) >= tolerance)
        return 2 * super().loss(beta, gamma, **kwargs) + (p + q) * np.log(
            self._jones2010n_eff())


    def muller_hui_2016ic(self, beta, gamma, tolerance=0., **kwargs):
        """
        Implements Information Criterion (IC) from (Muller, 2018)

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        tolerance : float, positive
            Threshold for absolute values of beta and gamma being considered zero.
            Should account for the finite tolerance of the numerical solver.

        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        Value of Mueller's IC
        """
        self._recalculate_cholesky(gamma)
        N = self.problem.num_obs
        # n_eff = self._jones2010n_eff()
        m = self.problem.num_groups
        return 2 / N * super().loss(beta, gamma, **kwargs) \
               + 1 / N * np.log(m) * sum(np.abs(beta) >= tolerance) \
               + 2 / N * sum(np.abs(gamma) >= tolerance)

    def vaida2005aic(self, beta, gamma, tolerance=0., marginalized=False, **kwargs):
        """
        Calculates Akaike Information Criterion (AIC) from https://www.jstor.org/stable/2673485

        Parameters
        ----------
        beta : np.ndarray, shape = [p]
            Vector of estimates of fixed effects.
        gamma : np.ndarray, shape = [q]
            Vector of estimates of random effects.
        tolerance : float, positive
            Threshold for absolute values of beta and gamma being considered zero.
            Should account for the finite tolerance of the numerical solver.

        kwargs :
            Not used, left for future and for passing debug/experimental parameters

        Returns
        -------
        Value for Vaida AIC
        """
        n = self.problem.num_obs
        p = sum(np.abs(beta) >= tolerance)
        q = sum(np.abs(gamma) >= tolerance)
        if marginalized:
            # mAIC version
            # See also p 141 eq 10 here
            # https://projecteuclid.org/journals/statistical-science/volume-28/issue-2/Model-Selection-in-Linear-Mixed-Models/10.1214/12-STS410.short
            if (n - p - q - 1) > 0:
                alpha = n / (n - p - q - 1)
            else:
                alpha = 1
            return 2 * super().loss(beta, gamma, **kwargs) + 2 * alpha * (p + q)
        else:
            # cAIC version
            # See also p 143 eq 17 here
            # https://projecteuclid.org/journals/statistical-science/volume-28/issue-2/Model-Selection-in-Linear-Mixed-Models/10.1214/12-STS410.short
            rho = self._hodges2001ddf(gamma)
            alpha = 2 * n / (n - p - 2) * (rho - (rho - p) / (n - p))
            return 2 * super().demarginalized_loss(beta, gamma, **kwargs) + alpha * (p + q)
