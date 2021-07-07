from typing import Tuple, Dict

import numpy as np

from .lme.problems import LinearLMEProblem


class GaussianPrior:
    def __init__(self, fe_params: Dict, re_params: Dict):
        self.fe_params = fe_params
        self.re_params = re_params
        self.fe_means = None
        self.fe_stds = None
        self.fe_weights = None
        self.re_means = None
        self.re_stds = None
        self.re_weights = None

    def instantiate(self, problem: LinearLMEProblem):
        assert problem.fe_columns and problem.re_columns, "Problem does not have column names attached"
        assert all(key in problem.fe_columns for key in self.fe_params.keys()), \
            F"Some keys are listed in the prior for FE but not listed in the prolem's column labels: {[key for key in self.fe_params.keys() if key not in problem.fe_columns]}"
        assert all(key in problem.fe_columns for key in self.fe_params.keys()), \
            F"Some keys are listed in the prior for RE but not listed in the prolem's column labels: {[key for key in self.re_params.keys() if key not in problem.re_columns]}"

        fe_means = []
        fe_stds = []
        fe_weights = []
        for label in problem.fe_columns:
            mean, std = self.fe_params.get(label, (0, 0))
            assert std >= 0
            fe_means.append(mean)
            fe_weights.append(1 if std > 0 else 0)
            fe_stds.append(std if std > 0 else 1)
        self.fe_means = np.array(fe_means)
        self.fe_stds = np.array(fe_stds)
        self.fe_weights = np.array(fe_weights)

        re_means = []
        re_stds = []
        re_weights = []
        for label in problem.re_columns:
            mean, std = self.re_params.get(label, (0, 0))
            assert std >= 0
            re_means.append(mean)
            re_weights.append(1 if std > 0 else 0)
            re_stds.append(std if std > 0 else 1)
        self.re_means = np.array(re_means)
        self.re_stds = np.array(re_stds)
        self.re_weights = np.array(re_weights)

    def forget(self):
        self.fe_means = None
        self.fe_stds = None
        self.fe_weights = None
        self.re_means = None
        self.re_stds = None
        self.re_weights = None

    def loss(self, beta, gamma):
        return (self.fe_weights * (1 / (2 * self.fe_stds)) * ((beta - self.fe_means) ** 2)).sum() + \
               (self.re_weights * (1 / (2 * self.re_stds)) * ((gamma - self.re_means) ** 2)).sum()

    def gradient_beta(self, beta, gamma):
        return self.fe_weights * (1 / self.fe_stds) * (beta - self.fe_means)

    def gradient_gamma(self, beta, gamma):
        return self.re_weights * (1 / self.re_stds) * (gamma - self.re_means)

    def hessian_beta(self, beta, gamma):
        return np.diag(self.fe_weights * (1 / self.fe_stds))

    def hessian_gamma(self, beta, gamma):
        return np.diag(self.re_weights * (1 / self.re_stds))

    def hessian_beta_gamma(self, beta, gamma):
        return 0


class NonInformativePrior:
    def __init__(self):
        pass

    def instantiate(self, problem):
        pass

    def forget(self):
        pass

    def loss(self, beta, gamma):
        return 0

    def gradient_beta(self, beta, gamma):
        return 0

    def gradient_gamma(self, beta, gamma):
        return 0

    def hessian_beta(self, beta, gamma):
        return 0

    def hessian_gamma(self, beta, gamma):
        return 0

    def hessian_beta_gamma(self, beta, gamma):
        return 0
