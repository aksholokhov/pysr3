from skmixed.lme.oracles import LinearLMEOracle, LinearLMEOracleSR3
import numpy as np
import scipy as sp


class PGDSolver:
    def __init__(self, tol=1e-4, max_iter=1000, stepping="fixed", fixed_step_len=1, **kwargs):
        self.tol = tol
        self.max_iter = max_iter
        self.stepping = stepping
        self.fixed_step_len = fixed_step_len

    def optimize(self, x0, oracle: LinearLMEOracle = None, regularizer=None, logger=None):
        if not oracle:
            raise ValueError("oracle can't be None")
        x = x0
        x_prev = np.infty
        iteration = 0

        if len(logger.keys) > 0:
            loss = oracle.value_function(x) + regularizer.value(x)
            logger.log(locals())

        while np.linalg.norm(x - x_prev) > self.tol and iteration < self.max_iter:
            x_prev = x

            direction = -oracle.gradient_value_function(x)
            # make sure gamma >= 0
            ind_zero_x = np.where((x <= 0) & (direction < 0))[0]
            ind_zero_x = ind_zero_x[(ind_zero_x >= oracle.problem.num_fixed_effects)]
            direction[ind_zero_x] = 0

            ind_neg_dir = np.where(direction < 0.0)[0]
            ind_neg_dir = ind_neg_dir[(ind_neg_dir >= oracle.problem.num_fixed_effects)]
            max_step_len = 1 if len(ind_neg_dir) == 0 else np.min(-x[ind_neg_dir] / direction[ind_neg_dir])

            if self.stepping == "line-search":
                res = sp.optimize.minimize(
                    fun=lambda t: oracle.value_function(regularizer.prox(x + t * direction, t)) + regularizer.value(
                        regularizer.prox(x + t * direction, t)),
                    x0=0,
                    bounds=[(0, max_step_len)]
                )
                step_len = res.x
            elif self.stepping == "fixed_max":
                step_len = min(max_step_len, self.fixed_step_len)
            else:
                step_len = self.fixed_step_len

            x = x + step_len * direction
            x = regularizer.prox(x, step_len)
            iteration += 1
            if len(logger.keys) > 0:
                loss = oracle.value_function(x) + regularizer.value(x)
                logger.log(locals())

        if iteration == self.max_iter:
            pass
            # did not converge
            # raise Exception(f"Did not converge, increase max_iter (current = {self.max_iter})")
        return x


class FakePGDSolver:

    def __init__(self, fixed_step_len=1, update_prox_every=1):
        self.fixed_step_len = fixed_step_len
        self.update_prox_every = update_prox_every

    def optimize(self, x0, oracle: LinearLMEOracleSR3 = None, regularizer=None, logger=None, **kwargs):
        if not oracle:
            raise ValueError("oracle can't be None")
        if not regularizer:
            raise ValueError("regularizer can't be None")

        x = oracle.find_optimal_parameters(x0, regularizer=regularizer, prox_step_len=self.fixed_step_len,
                                           update_prox_every=self.update_prox_every,
                                           **kwargs)

        if len(logger.keys) > 0:
            loss = oracle.value_function(x) + regularizer.value(x)
            logger.log(locals())

        return x


class AcceleratedPGDSolver:
    def __init__(self, tol=1e-4, max_iter=1000, stepping="fixed", fixed_step_len=1, **kwargs):
        self.tol = tol
        self.max_iter = max_iter
        self.stepping = stepping
        self.fixed_step_len = fixed_step_len

    def optimize(self, x0, oracle: LinearLMEOracle = None, regularizer=None, logger=None):
        if not oracle:
            raise ValueError("oracle can't be None")
        x = x0
        x_prev = np.infty
        iteration = 1

        if len(logger.keys) > 0:
            loss = oracle.value_function(x) + regularizer.value(x)
            logger.log(locals())

        while np.linalg.norm(x - x_prev) > self.tol and iteration <= self.max_iter:
            x_prev = x
            w = 0 if iteration == 1 else iteration / (iteration + 3)
            y = x + w * (x - x_prev)
            direction = -oracle.gradient_value_function(y)
            # make sure gamma >= 0
            ind_zero_y = np.where((y <= 0) & (direction < 0))[0]
            ind_zero_y = ind_zero_y[(ind_zero_y >= oracle.problem.num_fixed_effects)]
            direction[ind_zero_y] = 0

            ind_neg_dir = np.where(direction < 0.0)[0]
            ind_neg_dir = ind_neg_dir[(ind_neg_dir >= oracle.problem.num_fixed_effects)]
            max_step_len = 1 if len(ind_neg_dir) == 0 else np.min(-y[ind_neg_dir] / direction[ind_neg_dir])

            if self.stepping == "line-search":
                res = sp.optimize.minimize(
                    fun=lambda t: oracle.value_function(regularizer.prox(y + t * direction, t)) + regularizer.value(
                        regularizer.prox(y + t * direction, t)),
                    x0=0,
                    bounds=[(0, max_step_len)]
                )
                step_len = res.x
            elif self.stepping == "decreasing":
                step_len = max_step_len / (iteration + 1)
            else:
                step_len = self.fixed_step_len

            y = y + step_len * direction
            x = regularizer.prox(y, step_len)
            iteration += 1
            if len(logger.keys) > 0:
                loss = oracle.value_function(x) + regularizer.value(x)
                logger.log(locals())

        if iteration == self.max_iter:
            pass
            # did not converge
            # raise Exception(f"Did not converge, increase max_iter (current = {self.max_iter})")
        return x


class Fista:
    def __init__(self, tol=1e-4, max_iter=1000, stepping="fixed", fixed_step_len=1, **kwargs):
        self.tol = tol
        self.max_iter = max_iter
        self.stepping = stepping
        self.fixed_step_len = fixed_step_len

    def optimize(self, x0, oracle: LinearLMEOracle = None, regularizer=None, logger=None):
        if not oracle:
            raise ValueError("oracle can't be None")
        x = x0
        x_prev = np.infty
        iteration = 0
        a = 1
        y = x0

        if len(logger.keys) > 0:
            loss = oracle.value_function(x) + regularizer.value(x)
            logger.log(locals())

        while np.linalg.norm(x - x_prev) > self.tol and iteration < self.max_iter:
            x_prev = x

            direction = -oracle.gradient_value_function(x)
            # make sure gamma >= 0
            ind_zero_x = np.where((x <= 0) & (direction < 0))[0]
            ind_zero_x = ind_zero_x[(ind_zero_x >= oracle.problem.num_fixed_effects)]
            direction[ind_zero_x] = 0

            ind_neg_dir = np.where(direction < 0.0)[0]
            ind_neg_dir = ind_neg_dir[(ind_neg_dir >= oracle.problem.num_fixed_effects)]
            max_step_len = 1 if len(ind_neg_dir) == 0 else np.min(-x[ind_neg_dir] / direction[ind_neg_dir])

            if self.stepping == "decreasing":
                step_len = max_step_len / (iteration + 1)
            else:
                step_len = self.fixed_step_len

            y_next = regularizer.prox(x + step_len * direction, step_len)
            a_next = (1 + np.sqrt(1 + 4*a**2))/2
            x = y_next + (a - 1) / a_next * (y_next - y)

            y = y_next
            a = a_next

            iteration += 1
            if len(logger.keys) > 0:
                loss = oracle.value_function(x) + regularizer.value(x)
                logger.log(locals())

        if iteration == self.max_iter:
            pass
            # did not converge
            # raise Exception(f"Did not converge, increase max_iter (current = {self.max_iter})")
        return x

