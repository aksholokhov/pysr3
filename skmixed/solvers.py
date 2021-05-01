from skmixed.lme.oracles import LinearLMEOracle
import numpy as np


class PGDSolver:
    def __init__(self, tol=1e-4, max_iter=1000, stepping="fixed"):
        self.tol = tol
        self.max_iter = max_iter
        self.stepping = stepping

    def optimize(self, x0, oracle: LinearLMEOracle = None, regularizer=None, logger=None):
        if not oracle:
            raise ValueError("oracle can't be None")
        x = x0
        x_prev = 2 * (x0 + 1)
        iteration = 0

        if self.stepping == "fixed":
            step_size = 1
        else:
            raise ValueError(f"stepping is unknown: {self.stepping}")

        while np.linalg.norm(x - x_prev) > self.tol and iteration < self.max_iter:
            x_prev = x

            direction = -oracle.gradient_value_function(x)
            # make sure gamma >= 0
            ind_neg_dir = np.where(direction < 0.0)[0]
            ind_neg_dir = ind_neg_dir[(ind_neg_dir >= oracle.problem.num_fixed_effects)]
            max_step_len = min(step_size, 1 if len(ind_neg_dir) == 0 else np.min(-x[ind_neg_dir] / direction[ind_neg_dir]))
            max_step_len = 0.99*max_step_len/(iteration+1)

            x = x + max_step_len * direction
            x = regularizer.prox(x, step_size)
            iteration += 1
            if len(logger.keys) > 0:
                loss = oracle.value_function(x)
                logger.log(locals())

        if iteration == self.max_iter:
            pass
            # did not converge
            #raise Exception(f"Did not converge, increase max_iter (current = {self.max_iter})")
        return x
