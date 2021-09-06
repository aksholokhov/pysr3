"""
Implements general purpose numerical solvers, like PGD
"""

import numpy as np

from pysr3.lme.oracles import LinearLMEOracle
from pysr3.logger import Logger
from pysr3.regularizers import Regularizer


class PGDSolver:
    """
    Implements a general Proximal Gradient Descent solver.
    """

    def __init__(self, tol=1e-4, max_iter=1000, stepping="fixed", fixed_step_len=1):
        """
        Creates an instance of the solver.

        Parameters
        ----------
        tol: float
            Tolerance for the stop-criterion: norm(x - x0) is less than tol.
        max_iter: int
            Maximum number of iterations that the solver is allowed to make.
        stepping: str
            Stepping policy. Can be either "line-search" or "fixed".
        fixed_step_len: float
            Length of the step size. If stepping="fixed" then this step-size is always used.
            If stepping="line-search" then the line-search starts shrinking the step from this step size.
        """
        self.tol = tol
        self.max_iter = max_iter
        self.stepping = stepping
        self.fixed_step_len = fixed_step_len

    def optimize(self, x0, oracle: LinearLMEOracle = None, regularizer: Regularizer = None, logger: Logger = None):
        """
        Solves the optimization problem for

        Loss(x) = oracle(x) + regularizer(x)

        Parameters
        ----------
        x0: ndarray
            starting point of the optimizer.
        oracle: LinearLMEOracle
            provides the value and the gradient of the smooth part of the loss.
        regularizer: Regularizer
            provides the value and the proximal operator of the non-smooth part of the loss.
        logger: Logger
            logs the progress (loss, convergence, etc).

        Returns
        -------
        x: ndarray
            the minimum.
        """
        if not oracle:
            raise ValueError("oracle can't be None")
        x = x0
        x_prev = np.infty
        iteration = 0

        if 'loss' in logger.keys:
            loss = oracle.value_function(x) + regularizer.value(x)

        if len(logger.keys) > 0:
            logger.log(locals())

        while np.linalg.norm(x - x_prev) > self.tol and iteration < self.max_iter:
            x_prev = x

            direction = -oracle.gradient_value_function(x)

            if self.stepping == "line-search":
                step_len = self.fixed_step_len
                while step_len > 1e-14:
                    y = x + step_len * direction
                    z = regularizer.prox(y, step_len)
                    if oracle.value_function(z) <= oracle.value_function(x) - direction.dot(z - x) + (
                            1 / (2 * step_len)) * np.linalg.norm(z - x) ** 2:
                        break
                    else:
                        step_len *= 0.5

            elif self.stepping == "fixed":
                step_len = self.fixed_step_len
            else:
                step_len = self.fixed_step_len

            y = x + step_len * direction
            x = regularizer.prox(y, step_len)
            iteration += 1

            if 'loss' in logger.keys:
                loss = oracle.value_function(x) + regularizer.value(x)

            if len(logger.keys) > 0:
                logger.log(locals())

        logger.add("converged", iteration < self.max_iter)
        logger.add("iteration", iteration)

        return x


class FakePGDSolver:
    """
    This class is designed for the situations where the oracle can provide the optimal
    solution by itself, e.g. when it's accessible analytically.
    It's also used for PracticalSR3 methods, when the relaxed variables are
    updated together with the original ones inside of the oracle's subroutine.
    """

    def __init__(self, tol=1e-4, max_iter=1000, fixed_step_len=1, update_prox_every=1):
        """
        Initializes the solver

        Parameters
        ----------
        fixed_step_len: float
            step-size
        update_prox_every: int
            how often should the oracle update the relaxed variable (every X steps).
        """
        self.fixed_step_len = fixed_step_len
        self.update_prox_every = update_prox_every
        self.tol = tol
        self.max_iter = max_iter

    def optimize(self, x0, oracle=None, regularizer: Regularizer = None, logger: Logger = None,
                 **kwargs):
        """
        Solves the optimization problem for

        Loss(x) = oracle(x) + regularizer(x)

        Parameters
        ----------
        x0: ndarray
            starting point of the optimizer.
        oracle: LinearLMEOracle
            provides the value and the gradient of the smooth part of the loss.
        regularizer: Regularizer
            provides the value and the proximal operator of the non-smooth part of the loss.
        logger: Logger
            logs the progress (loss, convergence, etc).

        Returns
        -------
        x: ndarray
            the minimum.
        """
        if not oracle:
            raise ValueError("oracle can't be None")
        if not regularizer:
            raise ValueError("regularizer can't be None")

        x = oracle.find_optimal_parameters(x0,
                                           regularizer=regularizer,
                                           tol=self.tol,
                                           max_iter=self.max_iter,
                                           prox_step_len=self.fixed_step_len,
                                           update_prox_every=self.update_prox_every,
                                           logger=logger,
                                           **kwargs)
        if 'loss' in logger.keys:
            loss = oracle.value_function(x) + regularizer.value(x)

        if len(logger.keys) > 0:
            logger.log(locals())

        logger.add("converged", True)
        return x
