# This code implements solvers for linear mixed-effects models.
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


from typing import Set

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from skmixed.logger import Logger
from skmixed.lme.oracles import LinearLMEOracle

# TODO: implement sigma inference


class LMECascadeOracle:
    def __init__(self, levels, hierarchy):
        self.levels = levels
        self.hierarchy = hierarchy
        self.oracles = []
        for level in levels:
            self.oracles.append(LinearLMEOracle(level))

    def loss(self, beta, gamma):
        loss = 0
        for level, oracle in enumerate(self.oracles):
            loss += oracle.loss(beta, (level+1)*gamma)
        return loss

    def gradient_gamma(self, beta, gamma):
        gradient = np.zeros(len(gamma))
        for level, oracle in enumerate(self.oracles):
            gradient += (level+1)*oracle.gradient_gamma(beta, (level+1)*gamma)
        return gradient

    def optimal_beta(self, gamma):
        total_kernel = 0
        total_tail = 0
        for level, oracle in enumerate(self.oracles):
            kernel, tail = oracle.optimal_beta((level+1)*gamma, _dont_solve_wrt_beta=True)
            total_kernel += kernel
            total_tail += tail
        return np.linalg.solve(total_kernel, total_tail)

    def optimal_random_effects(self, beta, gamma):
        random_effects = {1: np.zeros(len(gamma))}
        for level, oracle in enumerate(self.oracles):
            for i, group in enumerate(oracle.problem.group_labels):
                parent = self.hierarchy[group]
                u = random_effects[parent]
                beta_current = beta.copy()
                for k in range(len(beta)):
                    j = int(oracle.beta_to_gamma_map[k])
                    if j >= 0:
                        beta_current[k] += u[j]
                random_effects[group] = oracle.optimal_random_effects(beta_current, gamma)[i] + u
        return random_effects


class LMECascade(BaseEstimator, RegressorMixin):
    def __init__(self,
                 tol: float = 1e-4,
                 tol_inner: float = 1e-4,
                 solver: str = "pgd",
                 n_iter: int = 1000,
                 n_iter_inner: int = 20,
                 use_line_search: bool = True,
                 logger_keys: Set = ('converged', 'loss',)):
        self.tol = tol
        self.tol_inner = tol_inner
        self.solver = solver
        self.n_iter = n_iter
        self.n_iter_inner = n_iter_inner
        self.use_line_search = use_line_search
        self.logger_keys = logger_keys

    def fit_problem(self, levels, hierarchy):
        oracle = LMECascadeOracle(levels, hierarchy)

        num_levels = len(levels)
        num_fixed_effects = levels[0].num_fixed_effects
        num_random_effects = levels[0].num_random_effects

        beta = np.ones(num_fixed_effects)
        gamma = 2*np.ones(num_random_effects)

        def projected_direction(current_gamma: np.ndarray, current_direction: np.ndarray):
            proj_direction = current_direction.copy()
            for j, _ in enumerate(current_gamma):
                if current_gamma[j] <= 1e-15 and current_direction[j] <= 0:
                    proj_direction[j] = 0
            return proj_direction

        loss = oracle.loss(beta, gamma)
        self.logger_ = Logger(self.logger_keys)

        prev_beta = np.infty
        prev_gamma = np.infty
        iteration = 0

        while (np.linalg.norm(beta - prev_beta) > self.tol
                    or np.linalg.norm(gamma - prev_gamma) > self.tol) and iteration < self.n_iter:

                if iteration >= self.n_iter:
                    us = oracle.optimal_random_effects(beta, gamma)
                    if len(self.logger_keys) > 0:
                        self.logger_.log(**locals())
                    self.coef_ = {"beta": beta,
                                  "gamma": gamma,
                                  "random_effects": us
                                  }
                    self.logger_.add("converged", 0)
                    return self

                if self.solver == 'pgd':

                    prev_beta = beta
                    prev_gamma = gamma

                    beta = oracle.optimal_beta(gamma)

                    # =============== INNER (GAMMA) ITERATION ===========
                    gradient_gamma = oracle.gradient_gamma(beta, gamma)
                    # projecting the gradient to the set of constraints
                    direction = projected_direction(gamma, -gradient_gamma)

                    inner_iteration = 0
                    while (np.linalg.norm(direction) > self.tol_inner
                           and inner_iteration < self.n_iter_inner):
                        if self.use_line_search:
                            # line search method
                            step_len = 0.1
                            for i, _ in enumerate(gamma):
                                if direction[i] < 0:
                                    step_len = min(-gamma[i] / direction[i], step_len)

                            current_loss = oracle.loss(beta, gamma)

                            while (oracle.loss(beta, gamma + step_len * direction)
                                   >= (1 - np.sign(current_loss) * 1e-5) * current_loss):
                                step_len *= 0.5
                                if step_len <= 1e-15:
                                    break
                        else:
                            # fixed step size
                            step_len = 1 / iteration
                        if step_len <= 1e-15:
                            break
                        gamma = gamma + step_len * direction
                        gradient_gamma = oracle.gradient_gamma(beta, gamma)
                        direction = projected_direction(gamma, -gradient_gamma)
                        inner_iteration += 1

                iteration += 1
                loss = oracle.loss(beta, gamma)
                if len(self.logger_keys) > 0:
                    self.logger_.log(locals())

        us = oracle.optimal_random_effects(beta, gamma)

        self.logger_.add('converged', 1)
        # self.logger_.add('iterations', iteration)

        self.coef_ = {
            "beta": beta,
            "gamma": gamma,
            "random_effects": us,
        }

        return self
