from typing import Set

import numpy as np

from skmixed.linear.oracles import LinearOracle, LinearOracleSR3
from skmixed.linear.problems import LinearProblem
from skmixed.logger import Logger
from skmixed.regularizers import L1Regularizer, CADRegularizer, SCADRegularizer, DummyRegularizer
from skmixed.solvers import PGDSolver, FakePGDSolver


class LinearModel:

    def __init__(self,
                 solver=None,
                 oracle=None,
                 regularizer=None,
                 logger_keys: Set = ('converged',)):
        """
        Initializes the model

        Parameters
        ----------
        solver: PGDSolver
            an instance of PGDSolver
        oracle: LinearLMEOracle
            an instance of LinearLMEOracle
        regularizer: Regularizer
            an instance of Regularizer
        logger_keys: Optional[Set[str]]
            list of values for the logger to track.
        """
        self.regularizer = regularizer
        self.oracle = oracle
        self.solver = solver
        self.logger_keys = logger_keys
        self.coef_ = None
        self.logger_ = None

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            initial_parameters: dict = None,
            warm_start=False,
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

        problem = LinearProblem.from_x_y(x=x, y=y)
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

        self.oracle.instantiate(problem)
        if self.regularizer:
            self.regularizer.instantiate(weights=problem.regularization_weights)

        if initial_parameters is None:
            initial_parameters = {}

        x = np.ones(problem.num_features)
        if warm_start:
            x = initial_parameters.get("x0", x)

        self.logger_ = Logger(self.logger_keys)

        optimal_x = self.solver.optimize(x, oracle=self.oracle, regularizer=self.regularizer, logger=self.logger_)

        self.coef_ = {
            "x": optimal_x,
        }
        self.oracle.forget()
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
        columns_labels : Optional[List[int]]
            List of column labels. There shall be only one column of group labels and answers STDs,
            and overall n columns with fixed effects (1 or 3) and k columns of random effects (2 or 3).

                - 1 : fixed effect
                - 2 : random effect
                - 3 : both fixed and random,
                - 0 : groups labels
                - 4 : answers standard deviations

        Returns
        -------
        y : np.ndarray
            Models predictions.
        """
        self.check_is_fitted()
        problem = LinearProblem.from_x_y(x, y=None)
        return self.predict_problem(problem, **kwargs)

    def predict_problem(self, problem, **kwargs):
        """
        Makes a prediction if .fit was called before and throws an error otherwise.

        Parameters
        ----------
        problem : LinearLMEProblem
            An instance of LinearLMEProblem. Should have the same format as the data
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

        x = self.coef_['x']

        assert problem.num_features == x.shape[0], \
            "Number of features is not the same to what it was in the train data."

        return problem.a.dot(x)

    def check_is_fitted(self):
        """
        Checks if the model was fitted before. Throws an error otherwise.

        Returns
        -------
        None
        """
        if not hasattr(self, "coef_") or self.coef_ is None:
            raise AssertionError("The model has not been fitted yet. Call .fit() first.")


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
        solver = PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                           fixed_step_len=5e-2 if not fixed_step_len else fixed_step_len)
        oracle = LinearOracle(None, prior=prior)
        regularizer = DummyRegularizer()
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         logger_keys=logger_keys)


class SimpleLinearModelSR3(LinearModel):
    def __init__(self,
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
        fixed_step_len = (1 if el == 0 else 1 / el) if not fixed_step_len else fixed_step_len
        if practical:
            solver = FakePGDSolver(tol=tol_solver, max_iter=max_iter_solver, fixed_step_len=fixed_step_len)
        else:
            solver = PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                               fixed_step_len=fixed_step_len)
        oracle = LinearOracleSR3(None, lam=el, prior=prior)
        regularizer = DummyRegularizer()
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         logger_keys=logger_keys)


class LinearL1Model(SimpleLinearModel):
    def __init__(self,
                 lam: float = 1,
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
        lam: float
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
        self.regularizer = L1Regularizer(lam=lam)


class LinearCADModel(SimpleLinearModel):
    def __init__(self,
                 lam: float = 1.,
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
        lam: float
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
        self.regularizer = CADRegularizer(lam=lam, rho=rho)


class LinearSCADModel(SimpleLinearModel):
    def __init__(self,
                 lam: float = 1.,
                 rho: float = 1.,
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
        lam: float
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
        self.regularizer = SCADRegularizer(lam=lam, rho=rho, sigma=sigma)


class LinearL1ModelSR3(SimpleLinearModelSR3):
    def __init__(self,
                 lam: float = 1.,
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
        lam: float
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
        self.regularizer = L1Regularizer(lam=lam)


class LinearCADModelSR3(SimpleLinearModelSR3):
    def __init__(self,
                 lam: float = 1.,
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
        lam: float
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
        self.regularizer = CADRegularizer(lam=lam, rho=rho)


class LinearSCADModelSR3(SimpleLinearModelSR3):
    def __init__(self,
                 lam: float = 1.,
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
        lam: float
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
        self.regularizer = SCADRegularizer(lam=lam, rho=rho, sigma=sigma)
