from skmixed.lme.oracles import LinearLMEOracle
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

        while np.linalg.norm(x - x_prev) > self.tol and iteration < self.max_iter:
            x_prev = x

            direction = -oracle.gradient_value_function(x)
            # make sure gamma >= 0
            ind_zero_x = np.where((x <= 0) & ( direction < 0))[0]
            ind_zero_x = ind_zero_x[(ind_zero_x >= oracle.problem.num_fixed_effects)]
            direction[ind_zero_x] = 0

            ind_neg_dir = np.where(direction < 0.0)[0]
            ind_neg_dir = ind_neg_dir[(ind_neg_dir >= oracle.problem.num_fixed_effects)]
            max_step_len = 1 if len(ind_neg_dir) == 0 else np.min(-x[ind_neg_dir] / direction[ind_neg_dir])

            if self.stepping == "line-search":
                res = sp.optimize.minimize(
                    fun=lambda t: oracle.value_function(regularizer.prox(x + t*direction, t)) + regularizer.value(regularizer.prox(x + t*direction, t)),
                    x0=0,
                    bounds=[(0, max_step_len)]
                )
                step_len = res.x
            elif self.stepping == "decreasing":
                step_len = max_step_len/(iteration+1)
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
            #raise Exception(f"Did not converge, increase max_iter (current = {self.max_iter})")
        return x
