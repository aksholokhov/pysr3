# This code implements solvers for linear mixed-effects models.
# Copyright (C) 2020 Aleksei Sholokhov, aksh@uw.edu
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
Linear Mixed-Effects Models (simple, relaxed, and regularized)
"""

import warnings
from typing import Set, Optional, Tuple, List

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, check_X_y, check_array
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from pysr3.lme.oracles import LinearLMEOracle, LinearLMEOracleSR3
from pysr3.lme.problems import LMEProblem
from pysr3.lme.problems import get_per_group_coefficients
from pysr3.logger import Logger
from pysr3.regularizers import Regularizer, L0Regularizer, L1Regularizer, CADRegularizer, SCADRegularizer, \
    DummyRegularizer, PositiveQuadrantRegularizer
from pysr3.solvers import PGDSolver, FakePGDSolver


class LMEModel(BaseEstimator, RegressorMixin):
    """
    Solve Linear Mixed Effects problem with projected gradient descent method.

    The original statistical model which this loss is based on is::

        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        u_i ~ 𝒩(0, diag(𝛄)),

        𝛄 ~ 𝒩(t𝛄, 1/lg)

        β ~ 𝒩(tβ, 1/lb)

        𝜺_i ~ 𝒩(0, diag(obs_var)

    See the paper for more details.
    """

    def __init__(self, logger_keys: Set = ('converged', 'iteration'), initializer: str = "None"):
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
        initializer: str
            "EM" or "None"
        logger_keys: Optional[Set[str]]
            list of values for the logger to track.
        """
        self.logger_keys = logger_keys
        self.initializer = initializer

    def instantiate(self) -> Tuple[Optional[LinearLMEOracle], Optional[Regularizer], Optional[PGDSolver]]:
        raise NotImplementedError("LinearModel is a base abstract class that should be used only for inheritance.")

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            columns_labels: np.ndarray = None,
            initial_parameters: dict = None,
            warm_start=False,
            fit_fixed_intercept=False,
            fit_random_intercept=False,
            fe_regularization_weights=None,
            re_regularization_weights=None,
            **kwargs):
        """
                Fits a Linear Model with Linear Mixed-Effects to the given data.

                Parameters
                ----------

                x : np.ndarray
                    Data. If columns_labels = None then it's assumed that columns_labels are in the first row of x.

                y : np.ndarray
                    Answers, real-valued array.

                columns_labels : np.ndarray
                    List of column labels. There shall be only one column of group labels and answers STDs,
                    and overall n columns with fixed effects (1 or 3) and k columns of random effects (2 or 3).

                        - 1 : fixed effect
                        - 2 : random effect
                        - 3 : both fixed and random,
                        - 0 : groups labels
                        - 4 : answers standard deviations

                initial_parameters : np.ndarray
                    Dict with possible fields:

                        -   | 'beta0' : np.ndarray, shape = [n],
                            | Initial estimate of fixed effects. If None then it defaults to an all-ones vector.
                        -   | 'gamma0' : np.ndarray, shape = [k],
                            | Initial estimate of random effects covariances.
                            | If None then it defaults to an all-ones vector.

                warm_start : bool, default is False
                    Whether to use previous parameters as initial ones. Overrides initial_parameters if given.
                    Throws NotFittedError if set to True when not fitted.

                fit_fixed_intercept : bool, default = False
                    Whether to add the intercept to the model

                fit_random_intercept : bool, default = False
                    Whether treat the intercept as a random effect.
                kwargs :
                    Not used currently, left here for passing debugging parameters.

                Returns
                -------
                self : LinearLMESparseModel
                    Fitted regression model.
                """
        check_X_y(x, y)
        x = np.array(x, dtype='float64')
        y = np.array(y, dtype='float64')
        if len(y.shape) > 1:
            warnings.warn("y with more than one dimension is not supported. First column taken.", DataConversionWarning)
            y = y[:, 0]

        problem = LMEProblem.from_x_y(x, y, columns_labels=columns_labels, fit_fixed_intercept=fit_fixed_intercept,
                                      fit_random_intercept=fit_random_intercept, **kwargs)
        return self.fit_problem(problem, initial_parameters=initial_parameters, warm_start=warm_start,
                                fe_regularization_weights=fe_regularization_weights,
                                re_regularization_weights=re_regularization_weights, **kwargs)

    def fit_problem(self,
                    problem: LMEProblem,
                    initial_parameters: dict = None,
                    warm_start=False,
                    fe_regularization_weights=None,
                    re_regularization_weights=None,
                    **kwargs):
        """
        Fits the model to a provided problem

        Parameters
        ----------
        problem: LMEProblem
            an instance of LinearLMEProblem that contains all data-dependent information

        initial_parameters : np.ndarray
            Dict with possible fields:

                -   | 'beta0' : np.ndarray, shape = [n],
                    | Initial estimate of fixed effects. If None then it defaults to an all-ones vector.
                -   | 'gamma0' : np.ndarray, shape = [k],
                    | Initial estimate of random effects covariances. If None then it defaults to an all-ones vector.

        warm_start : bool, default is False
            Whether to use previous parameters as initial ones. Overrides initial_parameters if given.
            Throws NotFittedError if set to True when not fitted.

        kwargs :
            Not used currently, left here for passing debugging parameters.

        Returns
        -------
            self
        """
        oracle, regularizer, solver = self.instantiate()

        oracle.instantiate(problem)
        if fe_regularization_weights is None:
            fe_regularization_weights = np.ones(problem.num_fixed_features)
        if re_regularization_weights is None:
            re_regularization_weights = np.ones(problem.num_random_features)
        regularizer.instantiate(weights=oracle.beta_gamma_to_x(beta=fe_regularization_weights,
                                                               gamma=re_regularization_weights),
                                oracle=oracle)

        num_fixed_effects = problem.num_fixed_features
        num_random_effects = problem.num_random_features

        if initial_parameters is None:
            initial_parameters = {}

        if hasattr(self, "coef_") and warm_start and check_is_fitted(self, 'coef_'):
            beta = self.coef_["beta"]
            gamma = self.coef_["gamma"]
        else:
            beta = initial_parameters.get("beta", np.ones(num_fixed_effects))
            gamma = initial_parameters.get("gamma", np.ones(num_random_effects))

        if self.initializer == "EM":
            beta = oracle.optimal_beta(gamma, tbeta=beta, beta=beta)
            us = oracle.optimal_random_effects(beta, gamma)
            gamma = np.sum(us ** 2, axis=0) / oracle.problem.num_groups

        self.logger_ = Logger(self.logger_keys)

        x = oracle.beta_gamma_to_x(beta, gamma)
        optimal_x = solver.optimize(x, oracle=oracle, regularizer=regularizer, logger=self.logger_)
        beta, gamma = oracle.x_to_beta_gamma(optimal_x)

        us = oracle.optimal_random_effects(beta, gamma)

        per_group_coefficients = get_per_group_coefficients(beta, us, labels=problem.column_labels)

        self.coef_ = {
            "beta": beta,
            "gamma": gamma,
            "random_effects": us,
            "group_labels": np.copy(problem.group_labels),
            "per_group_coefficients": per_group_coefficients,
        }

        if "vaida_aic" in self.logger_.keys:
            self.logger_.add("vaida_aic", oracle.vaida2005aic(beta, gamma, tolerance=np.sqrt(solver.tol)))
        if "vaida_aic_marginalized" in self.logger_.keys:
            self.logger_.add("vaida_aic_marginalized", oracle.vaida2005aic(beta, gamma,
                                                                           marginalized=True,
                                                                           tolerance=np.sqrt(solver.tol)))
        if "jones_bic" in self.logger_.keys:
            self.logger_.add("jones_bic", oracle.jones2010bic(beta, gamma, tolerance=np.sqrt(solver.tol)))
        if "muller_ic" in self.logger_.keys:
            self.logger_.add("muller_ic", oracle.muller_hui_2016ic(beta, gamma, tolerance=np.sqrt(solver.tol)))
        if "flip_probabilities_beta" in self.logger_.keys:
            self.logger_.add("flip_probabilities_beta", oracle.flip_probabilities_beta(beta, gamma))

        self.n_features_in_ = problem.num_features

        return self

    def predict(self, x, columns_labels: Optional[List[str]] = None,
                fit_fixed_intercept=False, fit_random_intercept=False,
                **kwargs):
        """
        Makes a prediction if .fit(X, y) was called before and throws an error otherwise.

        Parameters
        ----------
        x : np.ndarray
            Data matrix. Should have the same format as the data which was used for fitting the model:
            the number of columns and the columns' labels should be the same. It may contain new groups, in which case
            the prediction will be formed using the fixed effects only.

        columns_labels : Optional[List[str]]
            List of column labels. There shall be only one column of group labels and answers STDs,
            and overall n columns with fixed effects (1 or 3) and k columns of random effects (2 or 3).

                - 1 : fixed effect
                - 2 : random effect
                - 3 : both fixed and random,
                - 0 : groups labels
                - 4 : answers standard deviations

        fit_fixed_intercept: bool, default = True
            Whether to add an intercept as a fixed feature

        fit_random_intercept: bool, default = True
            Whether to add an intercept as a random feature.


        Returns
        -------
        y : np.ndarray
            Models predictions.
        """
        self.check_is_fitted()
        check_array(x)
        x = np.array(x)
        problem = LMEProblem.from_x_y(x, y=None, columns_labels=columns_labels, fit_fixed_intercept=fit_fixed_intercept,
                                      fit_random_intercept=fit_random_intercept)
        return self.predict_problem(problem, **kwargs)

    def predict_problem(self, problem, **kwargs):
        """
        Makes a prediction if .fit was called before and throws an error otherwise.

        Parameters
        ----------
        problem : LMEProblem
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

        beta = self.coef_['beta']
        us = self.coef_['random_effects']

        assert problem.num_fixed_features == beta.shape[0], \
            "Number of fixed effects is not the same to what it was in the train data."

        if len(us) > 0:
            assert problem.num_random_features == us[0].shape[0], \
                "Number of random effects is not the same to what it was in the train data."

        group_labels = self.coef_['group_labels']
        answers = []
        for i, (x, _, z, stds) in enumerate(problem):
            label = problem.group_labels[i]
            idx_of_this_label_in_train = np.where(group_labels == label)
            assert len(idx_of_this_label_in_train) <= 1, "Group labels of the classifier contain duplicates."
            if len(idx_of_this_label_in_train) == 1:
                idx_of_this_label_in_train = idx_of_this_label_in_train[0]
                y = x.dot(beta)
                if problem.num_random_features > 0:
                    y += z.dot(us[idx_of_this_label_in_train].flatten())
            else:
                # If we have not seen this group (so we don't have inferred random effects for this)
                # then we make a prediction with "expected" (e.g. zero) random effects
                y = x.dot(beta)
            answers.append(y)
        return np.concatenate(answers)

    def score(self, x, y, columns_labels=None, fit_fixed_intercept=False, fit_random_intercept=False,
              sample_weight=None):
        """
        Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum
        of squares ((y_true - y_pred) ** 2).sum() and v is the
        total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
        A constant model that always predicts the expected value of y,
        disregarding the input features, would get a R^2 score of 0.0.

        Parameters
        ----------
        x : np.ndarray
            Data matrix. Should have the same format as the data which was used for fitting the model:
            the number of columns and the columns' labels should be the same. It may contain new groups, in which case
            the prediction will be formed using the fixed effects only.

        y : np.ndarray
            Answers, real-valued array.

        columns_labels: np.ndarray
            Labels for columns of x

        sample_weight : array_like, Optional
            Weights of samples for calculating the R^2 statistics.

        Returns
        -------
        r2_score : float
            R^2 score

        """

        y_pred = self.predict(x, columns_labels=columns_labels,
                              fit_fixed_intercept=fit_fixed_intercept,
                              fit_random_intercept=fit_random_intercept, )
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v

    def check_is_fitted(self):
        """
        Checks if the model was fitted before. Throws an error otherwise.

        Returns
        -------
        None
        """
        if not hasattr(self, "coef_") or self.coef_ is None:
            raise NotFittedError("The model has not been fitted yet. Call .fit() first.")


class SimpleLMEModel(LMEModel):
    """
    Implements a standard Linear Mixed-Effects Model.
    """

    def __init__(self,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
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
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
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
        super().__init__(initializer=initializer, logger_keys=logger_keys)
        self.tol_solver = tol_solver
        self.max_iter_solver = max_iter_solver
        self.stepping = stepping
        self.logger_keys = logger_keys
        self.fixed_step_len = fixed_step_len
        self.prior = prior

    def instantiate(self):
        oracle = LinearLMEOracle(None, prior=self.prior)
        dummy_regularizer = DummyRegularizer()
        regularizer = PositiveQuadrantRegularizer(other_regularizer=dummy_regularizer)
        fixed_step_len = 5e-2 if not self.fixed_step_len else self.fixed_step_len
        solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                           fixed_step_len=fixed_step_len)

        return oracle, regularizer, solver

    def get_information_criterion(self, x, y, columns_labels=None, ic="muller_ic"):
        self.check_is_fitted()
        problem = LMEProblem.from_x_y(x, y, columns_labels=columns_labels)
        oracle = LinearLMEOracle(problem)
        oracle.instantiate(problem)
        if ic == "muller_ic":
            return oracle.muller_hui_2016ic(**self.coef_)
        elif ic == "vaida_aic":
            return oracle.vaida2005aic(**self.coef_)
        elif ic == "jones_bic":
            return oracle.jones2010bic(**self.coef_)
        else:
            raise ValueError(f"Unknown ic: {ic}")


class SimpleLMEModelSR3(LMEModel):
    """
    Implements Regularized Linear Mixed-Effects Model functional for given problem::

        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        β ~ 𝒩(tb, 1/lb),

        ||tβ||_0 = nnz(β) <= nnz_tbeta,

        u_i ~ 𝒩(0, diag(𝛄)),

        𝛄 ~ 𝒩(t𝛄, 1/lg),

        ||t𝛄||_0 = nnz(t𝛄) <= nnz_tgamma,

        𝜺_i ~ 𝒩(0, Λ)

    Here tβ and t𝛄 are single variables, not multiplications (e.g. not t*β). This oracle is designed for
    a solver (LinearLMESparseModel) which searches for a sparse solution (tβ, t𝛄) with at most k and j <= k non-zero
    elements respectively. For more details, see the documentation for LinearLMESparseModel.

    The problem should be provided as LMEProblem.

    """

    def __init__(self,
                 tol_oracle: float = 1e-5,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_oracle: int = 10000,
                 max_iter_solver: int = 10000,
                 stepping: str = "fixed",
                 ell: float = 40,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 update_prox_every=1,
                 fixed_step_len=None,
                 prior=None,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        tol_oracle: float
            tolerance for SR3 oracle's internal numerical subroutines
        tol_solver: float
            tolerance for the stop criterion of PGD solver
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        ell: float
            level of SR3-relaxation
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        warm_start_oracle: bool
            if fitting should be started from the current model's coefficients.
            Used for fine-tuning and iterative fitting.
        practical: bool
            whether to use SR3-Practical method, which works faster at the expense of accuracy
        update_prox_every: int
            how often update the relaxed variables. Only if practical=True
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
        super().__init__(initializer=initializer, logger_keys=logger_keys)
        self.tol_oracle = tol_oracle
        self.tol_solver = tol_solver
        self.max_iter_oracle = max_iter_oracle
        self.max_iter_solver = max_iter_solver
        self.stepping = stepping
        self.logger_keys = logger_keys
        self.fixed_step_len = fixed_step_len
        self.warm_start_oracle = warm_start_oracle
        self.prior = prior
        self.ell = ell
        self.practical = practical
        self.update_prox_every = update_prox_every

    def instantiate(self):
        if not self.fixed_step_len:
            fixed_step_len = 1 if self.ell == 0 else 1 / self.ell
        else:
            fixed_step_len = self.fixed_step_len
        if self.practical:
            solver = FakePGDSolver(update_prox_every=self.update_prox_every)
        else:
            solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                               fixed_step_len=fixed_step_len)
        oracle = LinearLMEOracleSR3(None, lb=self.ell, lg=self.ell, tol_inner=self.tol_oracle,
                                    n_iter_inner=self.max_iter_oracle,
                                    warm_start=self.warm_start_oracle, prior=self.prior)
        dummy_regularizer = DummyRegularizer()
        regularizer = PositiveQuadrantRegularizer(other_regularizer=dummy_regularizer)
        return oracle, regularizer, solver

    def get_information_criterion(self, x, y, columns_labels=None, ic="muller_ic"):
        self.check_is_fitted()
        problem = LMEProblem.from_x_y(x, y, columns_labels=columns_labels)
        oracle = LinearLMEOracleSR3(problem)
        oracle.instantiate(problem)
        if ic == "muller_ic":
            return oracle.muller_hui_2016ic(**self.coef_)
        elif ic == "vaida_aic":
            return oracle.vaida2005aic(**self.coef_)
        elif ic == "jones_bic":
            return oracle.jones2010bic(**self.coef_)
        else:
            raise ValueError(f"Unknown ic: {ic}")


class L0LmeModel(SimpleLMEModel):
    """
    Implements an L0-regularized Linear Mixed-Effect Model. It allows specifying the maximal number of
    non-zero fixed and random effects in your model
    """

    def __init__(self,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_solver: int = 10000,
                 stepping: str = "line-search",
                 nnz_tbeta: int = None,
                 nnz_tgamma: int = None,
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
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        nnz_tbeta : int
            the maximal number of non-zero fixed effects in your model
        nnz_tgamma : int
            the maximal number of non-zero random effects in your model
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
                         initializer=initializer,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         prior=prior)
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        l0_regularizer = L0Regularizer(nnz_tbeta=self.nnz_tbeta,
                                       nnz_tgamma=self.nnz_tgamma,
                                       oracle=oracle)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=l0_regularizer)
        return oracle, regularizer, solver


class L0LmeModelSR3(SimpleLMEModelSR3):
    """
    Implements Regularized Linear Mixed-Effects Model functional for given problem::

        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        β ~ 𝒩(tb, 1/lb),

        ||tβ||_0 = nnz(β) <= nnz_tbeta,

        u_i ~ 𝒩(0, diag(𝛄)),

        𝛄 ~ 𝒩(t𝛄, 1/lg),

        ||t𝛄||_0 = nnz(t𝛄) <= nnz_tgamma,

        𝜺_i ~ 𝒩(0, Λ)

    Here tβ and t𝛄 are single variables, not multiplications (e.g. not t*β). This oracle is designed for
    a solver (LinearLMESparseModel) which searches for a sparse solution (tβ, t𝛄) with at most k and j <= k non-zero
    elements respectively. For more details, see the documentation for LinearLMESparseModel.

    The problem should be provided as LMEProblem.

    """

    def __init__(self,
                 tol_oracle: float = 1e-5,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_oracle: int = 1000,
                 max_iter_solver: int = 1000,
                 stepping: str = "fixed",
                 ell: float = 40,
                 nnz_tbeta: int = 1,
                 nnz_tgamma: int = 1,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 update_prox_every=1,
                 fixed_step_len=None,
                 prior=None,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        tol_oracle: float
            tolerance for SR3 oracle's internal numerical subroutines
        tol_solver: float
            tolerance for the stop criterion of PGD solver
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        lb: float
            level of SR3-relaxation for fixed effects
        lg: float
            level of SR3-relaxation for random effects
        nnz_tbeta : int
            the maximal number of non-zero fixed effects in your model
        nnz_tgamma : int
            the maximal number of non-zero random effects in your model
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        warm_start_oracle: bool
            if fitting should be started from the current model's coefficients.
            Used for fine-tuning and iterative fitting.
        practical: bool
            whether to use SR3-Practical method, which works faster at the expense of accuracy
        update_prox_every: int
            how often update the relaxed variables. Only if practical=True
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

        super().__init__(tol_oracle=tol_oracle,
                         tol_solver=tol_solver,
                         initializer=initializer,
                         max_iter_oracle=max_iter_oracle,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         warm_start_oracle=warm_start_oracle,
                         practical=practical,
                         update_prox_every=update_prox_every,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         ell=ell,
                         prior=prior)
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        l0_regularizer = L0Regularizer(nnz_tbeta=self.nnz_tbeta,
                                       nnz_tgamma=self.nnz_tgamma,
                                       oracle=oracle)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=l0_regularizer)
        return oracle, regularizer, solver


class L1LmeModel(SimpleLMEModel):
    """
    Implements a LASSO-regularized Linear Mixed-Effect Model
    """

    def __init__(self,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_solver: int = 10000,
                 stepping: str = "line-search",
                 lam: float = 0,
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
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        lam: float
            strength of LASSO regularizer
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
                         initializer=initializer,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         prior=prior)
        self.lam = lam

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        fixed_step_len = 1 / (self.lam + 1) if not self.fixed_step_len else self.fixed_step_len
        solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                           fixed_step_len=fixed_step_len)
        l1_regularizer = L1Regularizer(lam=self.lam)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=l1_regularizer)
        return oracle, regularizer, solver


class L1LmeModelSR3(SimpleLMEModelSR3):
    """
    Implements an SR3-relaxed LASSO-regularized Linear Mixed-Effect Model
    """

    def __init__(self,
                 tol_oracle: float = 1e-5,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_oracle: int = 10000,
                 max_iter_solver: int = 10000,
                 stepping: str = "fixed",
                 ell: float = 40,
                 lam: float = 1,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 update_prox_every=1,
                 fixed_step_len=None,
                 prior=None,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        tol_oracle: float
            tolerance for SR3 oracle's internal numerical subroutines
        tol_solver: float
            tolerance for the stop criterion of PGD solver
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        ell: float
            level of SR3-relaxation
        lam: float
            strength of LASSO regularizer
        participation_in_selection: ndarray of int
            0 if the feature should be affected by the regularizer, 0 otherwise
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        warm_start_oracle: bool
            if fitting should be started from the current model's coefficients.
            Used for fine-tuning and iterative fitting.
        practical: bool
            whether to use SR3-Practical method, which works faster at the expense of accuracy
        update_prox_every: int
            how often update the relaxed variables. Only if practical=True
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

        super().__init__(tol_oracle=tol_oracle,
                         tol_solver=tol_solver,
                         initializer=initializer,
                         max_iter_oracle=max_iter_oracle,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         warm_start_oracle=warm_start_oracle,
                         practical=practical,
                         update_prox_every=update_prox_every,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         ell=ell,
                         prior=prior)
        self.lam = lam

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        l1_regularizer = L1Regularizer(lam=self.lam)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=l1_regularizer)
        return oracle, regularizer, solver


class CADLmeModel(SimpleLMEModel):
    """
    Implements a CAD-regularized Linear Mixed-Effects Model
    """

    def __init__(self,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_solver: int = 10000,
                 stepping: str = "line-search",
                 rho: float = 0.30,
                 lam: float = 1.0,
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
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        lam: float
            strength of CAD regularizer
        rho: float
            cut-off amplitude above which the coefficients are not penalized
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
                         initializer=initializer,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         prior=prior)
        self.lam = lam
        self.rho = rho

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        fixed_step_len = 1 / (self.lam + 1) if not self.fixed_step_len else self.fixed_step_len
        solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                           fixed_step_len=fixed_step_len)
        cad_regularizer = CADRegularizer(lam=self.lam, rho=self.rho)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=cad_regularizer)
        return oracle, regularizer, solver


class CADLmeModelSR3(SimpleLMEModelSR3):
    """
    Implements a CAD-regularized SR3-relaxed Linear Mixed-Effect Model
    """

    def __init__(self,
                 tol_oracle: float = 1e-5,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_oracle: int = 10000,
                 max_iter_solver: int = 10000,
                 stepping: str = "fixed",
                 ell: float = 40.0,
                 rho: float = 0.3,
                 lam: float = 1.0,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 update_prox_every=1,
                 fixed_step_len=None,
                 prior=None,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        tol_oracle: float
            tolerance for SR3 oracle's internal numerical subroutines
        tol_solver: float
            tolerance for the stop criterion of PGD solver
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        ell: float
            level of SR3-relaxation
        lam: float
            strength of CAD regularizer
        rho: float
            cut-off amplitude above which the coefficients are not penalized
        participation_in_selection: ndarray of int
            0 if the feature should be affected by the regularizer, 0 otherwise
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        warm_start_oracle: bool
            if fitting should be started from the current model's coefficients.
            Used for fine-tuning and iterative fitting.
        practical: bool
            whether to use SR3-Practical method, which works faster at the expense of accuracy
        update_prox_every: int
            how often update the relaxed variables. Only if practical=True
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

        super().__init__(tol_oracle=tol_oracle,
                         tol_solver=tol_solver,
                         initializer=initializer,
                         max_iter_oracle=max_iter_oracle,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         warm_start_oracle=warm_start_oracle,
                         practical=practical,
                         update_prox_every=update_prox_every,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         ell=ell,
                         prior=prior)
        self.lam = lam
        self.rho = rho

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        cad_regularizer = CADRegularizer(lam=self.lam, rho=self.rho)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=cad_regularizer)
        return oracle, regularizer, solver


class SCADLmeModel(SimpleLMEModel):
    """
    Implements SCAD-regularized Linear Mixed-Effect Model
    """

    def __init__(self,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_solver: int = 10000,
                 stepping: str = "line-search",
                 rho: float = 3.7,  # as per recommendation from (Fan, Li, 2001), p. 1351
                 sigma: float = 0.5,  # same
                 lam: float = 1.0,
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
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        lam: float
            strength of SCAD regularizer
        rho: float, rho > 1
            first knot of the SCAD spline
        sigma: float,
            a positive constant such that sigma*rho is the second knot of the SCAD spline
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
                         initializer=initializer,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         prior=prior)
        self.lam = lam
        self.rho = rho
        self.sigma = sigma

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        fixed_step_len = 1 / (self.lam + 1) if not self.fixed_step_len else self.fixed_step_len
        solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                           fixed_step_len=fixed_step_len)
        scad_regularizer = SCADRegularizer(lam=self.lam, rho=self.rho, sigma=self.sigma)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=scad_regularizer)
        return oracle, regularizer, solver


class SCADLmeModelSR3(SimpleLMEModelSR3):
    """
    Implements SCAD-regularized SR3-relaxed Linear Mixed-Effects Model
    """

    def __init__(self,
                 tol_oracle: float = 1e-5,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_oracle: int = 10000,
                 max_iter_solver: int = 10000,
                 stepping: str = "fixed",
                 ell: float = 40.0,
                 rho: float = 3.7,  # as per recommendation from (Fan, Li, 2001), p. 1351
                 sigma: float = 0.5,  # same
                 lam: float = 1.0,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 update_prox_every=1,
                 fixed_step_len=None,
                 prior=None,
                 **kwargs):
        """
        Initializes the model

        Parameters
        ----------
        tol_oracle: float
            tolerance for SR3 oracle's internal numerical subroutines
        tol_solver: float
            tolerance for the stop criterion of PGD solver
        initializer: str
            pre-initialization. Can be "None", in which case the algorithm starts with
            "all-ones" starting parameters, or "EM", in which case the algorithm does
            one step of EM algorithm
        max_iter_solver: int
            maximal number of iterations for PGD solver
        stepping: str
            step-size policy for PGD. Can be either "line-search" or "fixed"
        ell: float
            level of SR3-relaxation
        lam: float
            strength of SCAD regularizer
        rho: float, rho > 1
            first knot of the SCAD spline
        sigma: float, sigma > 1
            a positive constant such that sigma*rho is the second knot of the SCAD spline
        participation_in_selection: ndarray of int
            0 if the feature should be affected by the regularizer, 0 otherwise
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        warm_start_oracle: bool
            if fitting should be started from the current model's coefficients.
            Used for fine-tuning and iterative fitting.
        practical: bool
            whether to use SR3-Practical method, which works faster at the expense of accuracy
        update_prox_every: int
            how often update the relaxed variables. Only if practical=True
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

        super().__init__(tol_oracle=tol_oracle,
                         tol_solver=tol_solver,
                         initializer=initializer,
                         max_iter_oracle=max_iter_oracle,
                         max_iter_solver=max_iter_solver,
                         stepping=stepping,
                         warm_start_oracle=warm_start_oracle,
                         practical=practical,
                         update_prox_every=update_prox_every,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         ell=ell,
                         prior=prior)
        self.lam = lam
        self.rho = rho
        self.sigma = sigma

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        scad_regularizer = SCADRegularizer(lam=self.lam, rho=self.rho, sigma=self.sigma)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=scad_regularizer)
        return oracle, regularizer, solver


def _check_input_consistency(problem, beta=None, gamma=None, tbeta=None, tgamma=None):
    """
    Checks the consistency of .fit() arguments

    Parameters
    ----------
    problem : LMEProblem
        The problem which contains data
    beta : array-like, shape = [n], Optional
        Vector of fixed effects
    gamma : array-like, shape = [k], Optional
        Vector of random effects
    tbeta : array-like, shape = [n], Optional
        Vector of the sparse set of fixed effects (for regularized models)
    tgamma : array-like, shape = [k], Optional
        Vector of the sparse set of random effects (for regularized models)

    Returns
    -------
        output : None
            None if all the checks are passed, otherwise raises an exception
    """

    num_features = problem.num_fixed_features
    num_random_effects = problem.num_random_features
    if beta is not None:
        if tbeta is not None:
            check_consistent_length(beta, tbeta)
        assert len(beta) == num_features, "len(beta) is %d, but the number of features in data is %d" % (
            len(beta), num_features
        )
    if gamma is not None:
        if tgamma is not None:
            check_consistent_length(gamma, tgamma)
        assert len(gamma) == num_random_effects, "len(gamma) is %d, but the number of random effects in data is %d" % (
            len(gamma), num_random_effects
        )
    return None
