from typing import Set

from pysr3.linear.models import SimpleLinearModel, SimpleLinearModelSR3
from pysr3.linear.problems import LinearProblem
from pysr3.regularizers import DummyRegularizer, L1Regularizer
from pysr3.solvers import PGDSolver
from pysr3.glms.oracles import GLMOracle
from pysr3.glms.link_functions import PoissonLinkFunction


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
        oracle.instantiate(problem)
        if ic == "aic":
            return oracle.aic(**self.coef_)
        elif ic == "bic":
            return oracle.bic(**self.coef_)
        else:
            raise ValueError(f"Unknown ic: {ic}")


# class SimplePoissonModelSR3(SimpleLinearModelSR3):
#     def instantiate(self):
#         oracle, regularizer, solver = super().instantiate()
#         oracle = PossionOracleSR3(None, prior=self.prior)
#         return oracle, regularizer, solver
#
#     def get_information_criterion(self, x, y, ic="bic"):
#         self.check_is_fitted()
#         problem = LinearProblem.from_x_y(x, y)
#         oracle = PossionOracleSR3(problem)
#         oracle.instantiate(problem)
#         if ic == "aic":
#             return oracle.aic(**self.coef_)
#         elif ic == "bic":
#             return oracle.bic(**self.coef_)
#         else:
#             raise ValueError(f"Unknown ic: {ic}")


class PoissonL1Model(SimplePoissonModel):
    def __init__(self,
                 lam: float = 0,
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
        self.lam = lam

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        regularizer = L1Regularizer(lam=self.lam)
        return oracle, regularizer, solver


# class PoissonL1ModelSR3(SimplePoissonModelSR3):
#     def __init__(self,
#                  lam: float = 0.,
#                  el: float = 1.,
#                  tol_solver: float = 1e-5,
#                  max_iter_solver: int = 1000,
#                  stepping: str = "line-search",
#                  logger_keys: Set = ('converged',),
#                  fixed_step_len=None,
#                  prior=None,
#                  practical=False,
#                  **kwargs):
#         """
#         Initializes the model
#
#         Parameters
#         ----------
#         lam: float
#             strength of LASSO regularizer
#         el: float
#             constant for SR3 relaxation. Bigger values correspond to tighter relaxation.
#         tol_solver: float
#             tolerance for the stop criterion of PGD solver
#         max_iter_solver: int
#             maximal number of iterations for PGD solver
#         stepping: str
#             step-size policy for PGD. Can be either "line-search" or "fixed"
#         logger_keys: List[str]
#             list of keys for the parameters that the logger should track
#         fixed_step_len: float
#             step-size for PGD algorithm. If "linear-search" is used for stepping
#             then the algorithm uses this value as the maximal step possible. Use
#             this parameter if you know the Lipschitz-smoothness constant L for your problem
#             as fixed_step_len=1/L.
#         prior: Optional[Prior]
#             an instance of Prior class. If None then a non-informative prior is used.
#         kwargs:
#             for passing debugging info
#         """
#         super().__init__(el=el,
#                          tol_solver=tol_solver,
#                          max_iter_solver=max_iter_solver,
#                          stepping=stepping,
#                          logger_keys=logger_keys,
#                          fixed_step_len=fixed_step_len,
#                          prior=prior,
#                          practical=practical)
#         self.lam = lam
#
#     def instantiate(self):
#         oracle, regularizer, solver = super().instantiate()
#         regularizer = L1Regularizer(lam=self.lam)
#         return oracle, regularizer, solver