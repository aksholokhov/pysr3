"""
Prior distributions for models' parameters
"""

from typing import Dict

import numpy as np

from pysr3.lme.problems import LMEProblem


class Prior:
    pass


class GaussianPrior:
    """
        Implements Gaussian Prior for various models
        """

    def __init__(self, params: Dict):
        """
        Creates GaussianPrior

        Parameters
        ----------
        fe_params: dict[str: tuple(float, float)]
            gaussian prior parameters for fixed effects. The format is {"name": (mean, std), ...}
             E.g. {"intercept": (0, 2), "time": (1, 1)}
        params: dict[str: tuple(float, float)]
            gaussian prior for variances of random effects. Same format as above.
        """
        self.params = params
        self.means = None
        self.stds = None
        self.weights = None

    def instantiate(self, problem_columns):
        """
        Instantiates a Gaussian prior with problem-dependent quantities

        Parameters
        ----------
        problem: LMEProblem
            problem to fit

        Returns
        -------
        None
        """
        assert all(key in problem_columns for key in self.params.keys()), \
            (f"Some keys are listed in the prior but not listed in the prolem's column labels:" +
             f" {[key for key in self.params.keys() if key not in problem_columns]}")

        means = []
        stds = []
        weights = []
        for label in problem_columns:
            mean, std = self.params.get(label, (0, 0))
            assert std >= 0
            means.append(mean)
            weights.append(1 if std > 0 else 0)
            stds.append(std if std > 0 else 1)
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.weights = np.array(weights)

    def forget(self):
        """
        Releases all problem-dependent quantities

        Returns
        -------
        None
        """
        self.means = None
        self.stds = None
        self.weights = None

    def loss(self, x):
        """
        Value of the prior at beta, gamma.

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        value of the prior.
        """
        return (self.weights * (1 / (2 * self.stds)) * ((x - self.means) ** 2)).sum()

    def gradient(self, x):
        """
        Evaluates the gradient of the prior with respect to the vector of fixed effects

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        gradient w.r.t. beta
        """
        return self.weights * (1 / self.stds) * (x - self.means)

    def hessian(self, x):
        """
        Evaluates Hessian of the prior with respect to the vector of fixed effects

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        Hessian w.r.t. (beta, beta)
        """
        return np.diag(self.weights * (1 / self.stds))


class NonInformativePrior(Prior):
    """
    Implements a non-informative prior
    """

    def __init__(self):
        """
        Creates NonInformativePrior
        """
        pass

    def instantiate(self, problem):
        """
        Instantiates the prior based on the problem

        Parameters
        ----------
        problem: LMEProblem

        Returns
        -------
        None
        """
        pass

    def forget(self):
        """
        Releases all problem-dependent values

        Returns
        -------
        None
        """
        pass

    def loss(self, x):
        """
        Value of the prior at beta, gamma.

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        value of the prior.
        """
        return 0

    def gradient(self, x):
        """
        Evaluates the gradient of the prior with respect to the vector of fixed effects

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        gradient w.r.t. beta
        """
        return 0

    def hessian(self, x):
        """
        Evaluates Hessian of the prior with respect to the vector of random effects

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        Hessian w.r.t. (gamma, gamma)
        """
        return 0
