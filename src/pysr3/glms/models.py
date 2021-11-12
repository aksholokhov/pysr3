from typing import Set

import numpy as np

from pysr3.linear.models import SimpleLinearModel, SimpleLinearModelSR3
from pysr3.linear.problems import LinearProblem
from pysr3.glms.problems import PoissonProblem
from pysr3.regularizers import DummyRegularizer, L1Regularizer, PositiveQuadrantRegularizer, SCADRegularizer
from pysr3.solvers import PGDSolver
from pysr3.glms.oracles import GLMOracle, GLMOracleSR3
from pysr3.glms.link_functions import PoissonLinkFunction
from pysr3.preprocessors import Preprocessor


class SimplePoissonModel(SimpleLinearModel):

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        link_function = PoissonLinkFunction()
        oracle = GLMOracle(problem=None, prior=self.prior, link_function=link_function)
        return oracle, regularizer, solver

    def get_information_criterion(self, x, y, ic="bic"):
        self.check_is_fitted()
        problem = LinearProblem.from_x_y(x, y)
        link_function = PoissonLinkFunction()
        oracle = GLMOracle(problem=problem, prior=self.prior, link_function=link_function)
        if ic == "aic":
            return oracle.aic(**self.coef_)
        elif ic == "bic":
            return oracle.bic(**self.coef_)
        else:
            raise ValueError(f"Unknown ic: {ic}")

    def _get_tags(self):
        tags = super()._get_tags()
        tags["poor_score"] = True
        return tags

    def predict_problem(self, problem, **kwargs):
        """
        Makes a prediction if .fit was called before and throws an error otherwise.

        Parameters
        ----------
        problem :
            An instance of LMEProblem. Should have the same format as the data
            which was used for fitting the model. It may contain new groups, in which case
            the prediction will be formed using the fixed effects only.

        kwargs :
            for passing debugging parameters

        Returns
        -------
        y : np.ndarray
            Models predictions.
        """

        self.check_is_fitted()
        x = self.coef_
        problem_scaled, _ = Preprocessor.normalize(problem, **self.normalization_parameters_)
        if self.fit_intercept:
            problem_complete = Preprocessor.add_intercept(problem_scaled)
            x = np.concatenate([[self.intercept_], x])
        else:
            problem_complete = problem_scaled

        assert problem_complete.num_features == x.shape[0], \
            "Number of features is not the same to what it was in the train data."

        link_function = PoissonLinkFunction()

        return link_function.value(problem_complete.a.dot(x))


class SimplePoissonModelSR3(SimpleLinearModelSR3):
    def __init__(self,
                 constraints=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.constraints = constraints

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        link_function = PoissonLinkFunction()
        oracle = GLMOracleSR3(problem=None, prior=self.prior, lam=self.el, link_function=link_function, constraints=self.constraints)
        return oracle, regularizer, solver

    def _get_tags(self):
        tags = super()._get_tags()
        tags["poor_score"] = True
        return tags

    def get_information_criterion(self, x, y, ic="bic"):
        self.check_is_fitted()
        problem = LinearProblem.from_x_y(x, y)
        oracle = GLMOracleSR3(problem)
        oracle.instantiate(problem)
        if ic == "aic":
            return oracle.aic(**self.coef_)
        elif ic == "bic":
            return oracle.bic(**self.coef_)
        else:
            raise ValueError(f"Unknown ic: {ic}")

    def predict_problem(self, problem, **kwargs):
        """
        Makes a prediction if .fit was called before and throws an error otherwise.

        Parameters
        ----------
        problem :
            An instance of LMEProblem. Should have the same format as the data
            which was used for fitting the model. It may contain new groups, in which case
            the prediction will be formed using the fixed effects only.

        kwargs :
            for passing debugging parameters

        Returns
        -------
        y : np.ndarray
            Models predictions.
        """
        self.check_is_fitted()

        x = self.coef_
        problem_scaled, _ = Preprocessor.normalize(problem, **self.normalization_parameters_)
        if self.fit_intercept:
            problem_complete = Preprocessor.add_intercept(problem_scaled)
            x = np.concatenate([[self.intercept_], x])
        else:
            problem_complete = problem_scaled

        assert problem_complete.num_features == x.shape[0], \
            "Number of features is not the same to what it was in the train data."

        link_function = PoissonLinkFunction()

        return link_function.value(problem_complete.a.dot(x))


class PoissonL1Model(SimplePoissonModel):
    def __init__(self,
                 alpha: float = 0,
                 tol_solver: float = 1e-5,
                 max_iter_solver: int = 1000,
                 stepping: str = "line-search",
                 logger_keys: Set = ('converged',),
                 fixed_step_len=None,
                 prior=None,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        alpha: float
            strength of LASSO prior
        tol_solver: float
            tolerance for the stop criterion of PGD solver
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        fixed_step_len: float
            step-size for PGD algorithm. If "linear-search" is used for stepping
            then the algorithm uses this value as the maximal step possible. Use
            this parameter if you know the Lipschitz-smoothness constant L for your problem
            as fixed_step_len=1/L.
        prior: Optional[Prior]
            an instance of Prior class. If None then a non-informative prior is used.
        kwargs:
            for passing debugging info
        """
        super().__init__(tol_solver=tol_solver,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         prior=prior)
        self.alpha = alpha

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        regularizer = L1Regularizer(lam=self.alpha)
        return oracle, regularizer, solver


class PoissonL1ModelSR3(SimplePoissonModelSR3):
    def __init__(self,
                 alpha: float = 0.,
                 el: float = 1.,
                 tol_solver: float = 1e-5,
                 max_iter_solver: int = 1000,
                 stepping: str = "fixed",
                 logger_keys: Set = ('converged',),
                 fixed_step_len=None,
                 prior=None,
                 practical=False,
                 constraints=None,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        alpha: float
            strength of LASSO regularizer
        el: float
            constant for SR3 relaxation. Bigger values correspond to tighter relaxation.
        tol_solver: float
            tolerance for the stop criterion of PGD solver
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        fixed_step_len: float
            step-size for PGD algorithm. If "linear-search" is used for stepping
            then the algorithm uses this value as the maximal step possible. Use
            this parameter if you know the Lipschitz-smoothness constant L for your problem
            as fixed_step_len=1/L.
        prior: Optional[Prior]
            an instance of Prior class. If None then a non-informative prior is used.
        kwargs:
            for passing debugging info
        """
        super().__init__(el=el,
                         tol_solver=tol_solver,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         prior=prior,
                         practical=practical,
                         constraints=constraints,
                         **kwargs)
        self.alpha = alpha

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        regularizer = L1Regularizer(lam=self.alpha)
        return oracle, regularizer, solver



class PoissonSCADModelSR3(SimplePoissonModelSR3):
    def __init__(self,
                 alpha: float = 0.,
                 rho: float = 3.7,
                 sigma: float = 1.,
                 el: float = 1.,
                 tol_solver: float = 1e-5,
                 max_iter_solver: int = 1000,
                 stepping: str = "fixed",
                 logger_keys: Set = ('converged',),
                 fixed_step_len=None,
                 prior=None,
                 practical=False,
                 constraints=None,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        alpha: float
            strength of LASSO regularizer
        el: float
            constant for SR3 relaxation. Bigger values correspond to tighter relaxation.
        tol_solver: float
            tolerance for the stop criterion of PGD solver
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        fixed_step_len: float
            step-size for PGD algorithm. If "linear-search" is used for stepping
            then the algorithm uses this value as the maximal step possible. Use
            this parameter if you know the Lipschitz-smoothness constant L for your problem
            as fixed_step_len=1/L.
        prior: Optional[Prior]
            an instance of Prior class. If None then a non-informative prior is used.
        kwargs:
            for passing debugging info
        """
        super().__init__(el=el,
                         tol_solver=tol_solver,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         prior=prior,
                         practical=practical,
                         constraints=constraints,
                         **kwargs)
        self.alpha = alpha
        self.rho = rho
        self.sigma = sigma

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        regularizer = SCADRegularizer(lam=self.alpha, rho=self.rho, sigma=self.sigma)
        return oracle, regularizer, solver