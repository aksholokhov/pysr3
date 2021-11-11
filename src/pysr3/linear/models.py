import warnings
from typing import Set, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, check_X_y, check_array
from sklearn.exceptions import DataConversionWarning, NotFittedError

from pysr3.linear.oracles import LinearOracle, LinearOracleSR3
from pysr3.linear.problems import LinearProblem
from pysr3.logger import Logger
from pysr3.regularizers import L1Regularizer, CADRegularizer, SCADRegularizer, DummyRegularizer, Regularizer
from pysr3.solvers import PGDSolver, FakePGDSolver
from pysr3.preprocessors import Preprocessor


class LinearModel(BaseEstimator, RegressorMixin):

    def __init__(self, logger_keys=None, fit_intercept=True):
        """
        Initializes a linear model.

        Parameters
        ----------
        logger_keys: Tuple[str]
            Set of values that the logger is supposed to log
        """
        self.logger_keys = logger_keys
        self.fit_intercept = fit_intercept

    def instantiate(self) -> Tuple[Optional[LinearOracle], Optional[Regularizer], Optional[PGDSolver]]:
        raise NotImplementedError("LinearModel is a base abstract class that should be used only for inheritance.")

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            initial_parameters: dict = None,
            warm_start=False,
            regularization_weights=None,
            obs_std=None,
            sample_weight=None,
            **kwargs):
        """
        Fits a Linear Model to the given data.

        Parameters
        ----------
        x : np.ndarray
            Data

        y : np.ndarray
            Answers, real-valued array.

        initial_parameters : np.ndarray
            Dict with possible fields:

                -   | 'x0' : np.ndarray, shape = [n],
                    | Initial estimate of model's coefficients. If None then it defaults to an all-ones vector.

        warm_start : bool, default is False
            Whether to use previous parameters as initial ones. Overrides initial_parameters if given.
            Throws NotFittedError if set to True when not fitted.

        kwargs :
            Not used currently, left here for passing debugging parameters.

        Returns
        -------
        self : LinearLMESparseModel
            Fitted regression model.
        """
        check_X_y(x, y)
        if sample_weight is not None and type(sample_weight) == np.ndarray:
            if len(y) != len(sample_weight):
                raise ValueError("Sample weights should be the same length as y")
            if sample_weight.shape != y.shape:
                raise ValueError("Sample weights should be the same shape as y")

            idx_non_zero_sample_weights = np.where(sample_weight != 0)
            x = x[idx_non_zero_sample_weights]
            y = y[idx_non_zero_sample_weights]
            sample_weight = sample_weight[idx_non_zero_sample_weights]

        x = np.array(x)
        y = np.array(y)
        if len(y.shape) > 1:
            warnings.warn("y with more than one dimension is not supported. First column taken.", DataConversionWarning)
            y = y[:, 0]
        if obs_std is None:
            if sample_weight is not None and type(sample_weight) == np.ndarray:
                obs_std = 1/sample_weight
            else:
                obs_std = np.ones(len(y))
        problem = LinearProblem.from_x_y(x=x, y=y, obs_std=obs_std, regularization_weights=regularization_weights)
        return self.fit_problem(problem, initial_parameters=initial_parameters, warm_start=warm_start, **kwargs)

    def fit_problem(self,
                    problem: LinearProblem,
                    initial_parameters: dict = None,
                    warm_start=False,
                    **kwargs):
        """
        Fits the model to a provided problem

        Parameters
        ----------
        problem: LinearProblem
            an instance of LinearProblem that contains all data-dependent information

        initial_parameters : np.ndarray
            Dict with possible fields:

                -   | 'x0' : np.ndarray, shape = [n],
                    | Initial estimate of fixed effects. If None then it defaults to an all-ones vector.

        warm_start : bool, default is False
            Whether to use previous parameters as initial ones. Overrides initial_parameters if given.
            Throws NotFittedError if set to True when not fitted.

        kwargs :
            Not used currently, left here for passing debugging parameters.

        Returns
        -------
            self
        """
        normalized_problem, normalization_parameters = Preprocessor.normalize(problem)
        self.normalization_parameters_ = normalization_parameters
        if self.fit_intercept:
            problem_complete = Preprocessor.add_intercept(normalized_problem)
        else:
            problem_complete = normalized_problem

        oracle, regularizer, solver = self.instantiate()
        oracle.instantiate(problem_complete)
        regularizer.instantiate(weights=problem_complete.regularization_weights)

        if initial_parameters is None:
            initial_parameters = {}

        x = np.ones(problem_complete.num_features) / problem.num_features
        if warm_start:
            x = initial_parameters.get("x0", x)

        self.logger_ = Logger(self.logger_keys)

        optimal_x = solver.optimize(x, oracle=oracle, regularizer=regularizer, logger=self.logger_)

        if "iteration" in self.logger_.keys:
            self.n_iter_ = self.logger_.get("iteration")
        else:
            self.n_iter_ = 0

        if self.fit_intercept:
            self.intercept_ = optimal_x[0]
            self.coef_ = optimal_x[1:]
        else:
            self.intercept_ = 0

        if "aic" in self.logger_.keys:
            self.logger_.add("aic", oracle.aic(optimal_x))
        if "bic" in self.logger_.keys:
            self.logger_.add("bic", oracle.bic(optimal_x))
        self.n_features_in_ = problem.num_features

        return self

    def predict(self, x, **kwargs):
        """
        Makes a prediction if .fit(X, y) was called before and throws an error otherwise.

        Parameters
        ----------
        x : np.ndarray
            Data matrix. Should have the same format as the data which was used for fitting the model:
            the number of columns and the columns' labels should be the same. It may contain new groups, in which case
            the prediction will be formed using the fixed effects only.

        Returns
        -------
        y : np.ndarray
            Models predictions.
        """
        self.check_is_fitted()
        check_array(x)
        x = np.array(x)
        problem = LinearProblem.from_x_y(x, y=None)
        return self.predict_problem(problem, **kwargs)

    def predict_problem(self, problem, **kwargs):
        """
        Makes a prediction if .fit was called before and throws an error otherwise.

        Parameters
        ----------
        problem : LinearProblem
            An instance of LinearProblem. Should have the same format as the data
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

        return problem_complete.a.dot(x)

    def check_is_fitted(self):
        """
        Checks if the model was fitted before. Throws an error otherwise.

        Returns
        -------
        None
        """
        if not hasattr(self, "coef_") or self.coef_ is None:
            raise NotFittedError("The model has not been fitted yet. Call .fit() first.")


class SimpleLinearModel(LinearModel):
    def __init__(self,
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
        super().__init__()
        self.tol_solver = tol_solver
        self.max_iter_solver = max_iter_solver
        self.stepping = stepping
        self.logger_keys = logger_keys
        self.fixed_step_len = fixed_step_len
        self.prior = prior

    def instantiate(self):
        oracle = LinearOracle(None, prior=self.prior)
        regularizer = DummyRegularizer()
        solver = PGDSolver(tol=self.tol_solver,
                           max_iter=self.max_iter_solver,
                           stepping=self.stepping,
                           fixed_step_len=5e-2 if not self.fixed_step_len else self.fixed_step_len)
        return oracle, regularizer, solver

    def get_information_criterion(self, x, y, ic="bic"):
        self.check_is_fitted()
        problem = LinearProblem.from_x_y(x, y)
        oracle = LinearOracle(problem)
        oracle.instantiate(problem)
        if ic == "aic":
            return oracle.aic(**self.coef_)
        elif ic == "bic":
            return oracle.bic(**self.coef_)
        else:
            raise ValueError(f"Unknown ic: {ic}")


class SimpleLinearModelSR3(LinearModel):
    def __init__(self,
                 el: float = 1.,
                 tol_solver: float = 1e-5,
                 max_iter_solver: int = 1000,
                 stepping: str = "fixed",
                 logger_keys: Set = ('converged',),
                 fixed_step_len=None,
                 prior=None,
                 practical=False,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
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
        practical: bool
            whether to use a direct efficient value function evaluation method
        kwargs:
            for passing debugging info
        """
        super().__init__()
        self.el = el
        self.tol_solver = tol_solver
        self.max_iter_solver = max_iter_solver
        self.stepping = stepping
        self.logger_keys = logger_keys
        self.fixed_step_len = fixed_step_len
        self.practical = practical
        self.prior = prior

    def instantiate(self):
        fixed_step_len = (1 if self.el == 0 else 1 / self.el) if not self.fixed_step_len else self.fixed_step_len
        if self.practical:
            solver = FakePGDSolver(tol=self.tol_solver,
                                   max_iter=self.max_iter_solver,
                                   fixed_step_len=fixed_step_len)
        else:
            solver = PGDSolver(tol=self.tol_solver,
                               max_iter=self.max_iter_solver,
                               stepping=self.stepping,
                               fixed_step_len=fixed_step_len)
        oracle = LinearOracleSR3(None,
                                 lam=self.el,
                                 prior=self.prior)
        regularizer = DummyRegularizer()
        return oracle, regularizer, solver

    def get_information_criterion(self, x, y, ic="bic"):
        self.check_is_fitted()
        problem = LinearProblem.from_x_y(x, y)
        oracle = LinearOracleSR3(problem)
        oracle.instantiate(problem)
        if ic == "aic":
            return oracle.aic(**self.coef_)
        elif ic == "bic":
            return oracle.bic(**self.coef_)
        else:
            raise ValueError(f"Unknown ic: {ic}")


class LinearL1Model(SimpleLinearModel):
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


class LinearCADModel(SimpleLinearModel):
    def __init__(self,
                 alpha: float = 0.,
                 rho: float = 1.,
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
            strength of CAD regularizer
        rho: float
            cut-off amplitude above which the coefficients are not penalized
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
        self.rho = rho

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        regularizer = CADRegularizer(lam=self.alpha, rho=self.rho)
        return oracle, regularizer, solver


class LinearSCADModel(SimpleLinearModel):
    def __init__(self,
                 alpha: float = 0.,
                 rho: float = 3.7,
                 sigma: float = 1.,
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
            strength of SCAD regularizer
        rho: float, rho > 1
            first knot of the SCAD spline
        sigma: float,
            a positive constant such that sigma*rho is the second knot of the SCAD spline
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
        self.rho = rho
        self.sigma = sigma

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        regularizer = SCADRegularizer(lam=self.alpha, rho=self.rho, sigma=self.sigma)
        return oracle, regularizer, solver


class LinearL1ModelSR3(SimpleLinearModelSR3):
    def __init__(self,
                 alpha: float = 0.,
                 el: float = 1.,
                 tol_solver: float = 1e-5,
                 max_iter_solver: int = 1000,
                 stepping: str = "line-search",
                 logger_keys: Set = ('converged',),
                 fixed_step_len=None,
                 prior=None,
                 practical=False,
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
                         practical=practical)
        self.alpha = alpha

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        regularizer = L1Regularizer(lam=self.alpha)
        return oracle, regularizer, solver


class LinearCADModelSR3(SimpleLinearModelSR3):
    def __init__(self,
                 alpha: float = 0.,
                 rho: float = 1.,
                 el: float = 1.,
                 tol_solver: float = 1e-5,
                 max_iter_solver: int = 1000,
                 stepping: str = "line-search",
                 logger_keys: Set = ('converged',),
                 fixed_step_len=None,
                 prior=None,
                 practical=False,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        alpha: float
            strength of CAD regularizer
        rho: float
            cut-off amplitude above which the coefficients are not penalized
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
                         practical=practical)
        self.alpha = alpha
        self.rho = rho

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        regularizer = CADRegularizer(lam=self.alpha, rho=self.rho)
        return oracle, regularizer, solver


class LinearSCADModelSR3(SimpleLinearModelSR3):
    def __init__(self,
                 alpha: float = 0.,
                 rho: float = 2.,
                 sigma: float = 1.,
                 el: float = 1.,
                 tol_solver: float = 1e-5,
                 max_iter_solver: int = 1000,
                 stepping: str = "line-search",
                 logger_keys: Set = ('converged',),
                 fixed_step_len=None,
                 prior=None,
                 practical=False,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        alpha: float
            strength of SCAD regularizer
        rho: float, rho > 1
            first knot of the SCAD spline
        sigma: float,
            a positive constant such that sigma*rho is the second knot of the SCAD spline
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
                         practical=practical)
        self.alpha = alpha
        self.rho = rho
        self.sigma = sigma

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        regularizer = SCADRegularizer(lam=self.alpha, rho=self.rho, sigma=self.sigma)
        return oracle, regularizer, solver
