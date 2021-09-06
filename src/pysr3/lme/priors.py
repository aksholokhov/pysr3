from typing import Dict

from pysr3.lme.problems import LMEProblem
from pysr3.priors import Prior, GaussianPrior


class GaussianPriorLME:
    """
    Implements Gaussian Prior for various models
    """

    def __init__(self, fe_params: Dict, re_params: Dict):
        """
        Creates GaussianPrior

        Parameters
        ----------
        fe_params: dict[str: tuple(float, float)]
            gaussian prior parameters for fixed effects. The format is {"name": (mean, std), ...}
             E.g. {"intercept": (0, 2), "time": (1, 1)}
        re_params: dict[str: tuple(float, float)]
            gaussian prior for variances of random effects. Same format as above.
        """
        self.fe_params = fe_params
        self.re_params = re_params
        self.beta_prior = GaussianPrior(params=fe_params)
        self.gamma_prior = GaussianPrior(params=re_params)

    def instantiate(self, problem: LMEProblem):
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
        assert problem.fixed_features_columns and problem.random_features_columns, "Problem does not have column names attached"
        assert all(key in problem.fixed_features_columns for key in self.fe_params.keys()), \
            F"Some keys are listed in the prior for FE but not listed in the prolem's column labels: {[key for key in self.fe_params.keys() if key not in problem.fixed_features_columns]}"
        assert all(key in problem.fixed_features_columns for key in self.fe_params.keys()), \
            F"Some keys are listed in the prior for RE but not listed in the prolem's column labels: {[key for key in self.re_params.keys() if key not in problem.random_features_columns]}"

        self.beta_prior.instantiate(problem_columns=problem.fixed_features_columns)
        self.gamma_prior.instantiate(problem_columns=problem.random_features_columns)

    def forget(self):
        """
        Releases all problem-dependent quantities

        Returns
        -------
        None
        """
        self.fe_params = None
        self.re_params = None
        self.beta_prior.forget()
        self.gamma_prior.forget()

    def loss(self, beta, gamma):
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
        return self.beta_prior.loss(beta) + self.gamma_prior.loss(gamma)

    def gradient_beta(self, beta, *args, **kwargs):
        """
        Evaluates the gradient of the prior with respect to the vector of fixed effects

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        Returns
        -------
        gradient w.r.t. beta
        """
        return self.beta_prior.gradient(beta)

    def gradient_gamma(self, beta, gamma):
        """
        Evaluates the gradient of the prior with respect to the vector of random effects

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        gradient w.r.t. gamma
        """
        return self.gamma_prior.gradient(gamma)

    def hessian_beta(self, beta, gamma):
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
        return self.beta_prior.hessian(beta)

    def hessian_gamma(self, beta, gamma):
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
        return self.gamma_prior.hessian(gamma)

    def hessian_beta_gamma(self, beta, gamma):
        """
        Evaluates the mixed Hessian of the prior with respect to the vector of fixed and random effects

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        Hessian w.r.t. (beta, gamma)
        """
        return 0


class NonInformativePriorLME(Prior):
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

    def loss(self, beta, gamma):
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

    def gradient_beta(self, beta, gamma):
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

    def gradient_gamma(self, beta, gamma):
        """
        Evaluates the gradient of the prior with respect to the vector of random effects

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        gradient w.r.t. gamma
        """
        return 0

    def hessian_beta(self, beta, gamma):
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
        return 0

    def hessian_gamma(self, beta, gamma):
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

    def hessian_beta_gamma(self, beta, gamma):
        """
        Evaluates the mixed Hessian of the prior with respect to the vector of fixed and random effects

        Parameters
        ----------
        beta: ndarray
            vector of fixed effects

        gamma: ndarray
            vector of random effects

        Returns
        -------
        Hessian w.r.t. (beta, gamma)
        """
        return 0
