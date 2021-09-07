# Prior distributions for model parameters
# Copyright (C) 2021 Aleksei Sholokhov, aksh@uw.edu
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

"""
Prior distributions for model parameters
"""

from typing import Dict

import numpy as np


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
        problem_columns: List[str]
            Names of the columns for a particular dataset. Matches the elements of self.params (dict)
            with the columns of a particular dataset.

        Returns
        -------
        None
        """
        assert all(key in problem_columns for key in self.params.keys()), \
            (f"Some keys are listed in the prior but not listed in the problem's column labels:" +
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
        x: ndarray
            vector of parameters

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
        x: ndarray
            vector of parameters

        Returns
        -------
        gradient
        """
        return self.weights * (1 / self.stds) * (x - self.means)

    def hessian(self, _):
        """
        Evaluates Hessian of the prior with respect to the vector of fixed effects

        Returns
        -------
        Hessian
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

    @staticmethod
    def loss(_):
        """
        Value of the prior at beta, gamma.

        Returns
        -------
        value of the prior.
        """
        return 0

    @staticmethod
    def gradient(_):
        """
        Evaluates the gradient of the prior with respect to the vector of fixed effects

        Returns
        -------
        gradient
        """
        return 0

    @staticmethod
    def hessian(_):
        """
        Evaluates Hessian of the prior with respect to the vector of random effects

        Returns
        -------
        Hessian w.r.t. (gamma, gamma)
        """
        return 0
