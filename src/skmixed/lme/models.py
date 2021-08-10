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

from typing import Set

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted
from skmixed.helpers import get_per_group_coefficients
from skmixed.lme.oracles import LinearLMEOracle, LinearLMEOracleSR3
from skmixed.lme.problems import LinearLMEProblem
from skmixed.logger import Logger
from skmixed.regularizers import Regularizer, L0Regularizer, L1Regularizer, CADRegularizer, SCADRegularizer, \
    DummyRegularizer, PositiveQuadrantRegularizer
from skmixed.solvers import PGDSolver, FakePGDSolver


class LMEModel(BaseEstimator, RegressorMixin):
    """
    Solve Linear Mixed Effects problem with projected gradient descent method.

    The original statistical model which this loss is based on is::

        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        u_i ~ 𝒩(0, diag(𝛄)),

        𝛄 ~ 𝒩(t𝛄, 1/lg)

        β ~ 𝒩(tβ, 1/lb)

        𝜺_i ~ 𝒩(0, diag(obs_std)

    See the paper for more details.
    """

    def __init__(self,
                 solver=None,
                 oracle=None,
                 regularizer=None,
                 initializer: str = "None",
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
        initializer: str
            "EM" or "None"
        logger_keys: Optional[Set[str]]
            list of values for the logger to track.
        """
        self.regularizer = regularizer
        self.oracle = oracle
        self.solver = solver
        self.initializer = initializer
        self.logger_keys = logger_keys
        self.coef_ = None
        self.logger_ = None

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            columns_labels: np.ndarray = None,
            initial_parameters: dict = None,
            warm_start=False,
            random_intercept=True,
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

                random_intercept : bool, default = True
                    Whether treat the intercept as a random effect.
                kwargs :
                    Not used currently, left here for passing debugging parameters.

                Returns
                -------
                self : LinearLMESparseModel
                    Fitted regression model.
                """

        problem = LinearLMEProblem.from_x_y(x, y, columns_labels, random_intercept=random_intercept, **kwargs)
        return self.fit_problem(problem, initial_parameters=initial_parameters, warm_start=warm_start, **kwargs)

    def fit_problem(self,
                    problem: LinearLMEProblem,
                    initial_parameters: dict = None,
                    warm_start=False,
                    **kwargs):
        """
        Fits the model to a provided problem

        Parameters
        ----------
        problem: LinearLMEProblem
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

        self.oracle.instantiate(problem)
        if self.regularizer:
            self.regularizer.instantiate(weights=self.oracle.beta_gamma_to_x(beta=problem.fe_regularization_weights,
                                                                             gamma=problem.re_regularization_weights),
                                         oracle=self.oracle)

        num_fixed_effects = problem.num_fixed_effects
        num_random_effects = problem.num_random_effects

        if initial_parameters is None:
            initial_parameters = {}

        if hasattr(self, "coef_") and warm_start and check_is_fitted(self, 'coef_'):
            beta = self.coef_["beta"]
            gamma = self.coef_["gamma"]
        else:
            beta = initial_parameters.get("beta", np.ones(num_fixed_effects))
            gamma = initial_parameters.get("gamma", np.ones(num_random_effects))

        if self.initializer == "EM":
            beta = self.oracle.optimal_beta(gamma, tbeta=beta, beta=beta)
            us = self.oracle.optimal_random_effects(beta, gamma)
            gamma = np.sum(us ** 2, axis=0) / self.oracle.problem.num_groups

        self.logger_ = Logger(self.logger_keys)

        x = self.oracle.beta_gamma_to_x(beta, gamma)
        optimal_x = self.solver.optimize(x, oracle=self.oracle, regularizer=self.regularizer, logger=self.logger_)
        beta, gamma = self.oracle.x_to_beta_gamma(optimal_x)

        us = self.oracle.optimal_random_effects(beta, gamma)

        per_group_coefficients = get_per_group_coefficients(beta, us, labels=problem.column_labels)

        self.coef_ = {
            "beta": beta,
            "gamma": gamma,
            "random_effects": us,
            "group_labels": np.copy(problem.group_labels),
            "per_group_coefficients": per_group_coefficients,
        }

        # self.oracle.forget()

        return self

    def predict(self, x, columns_labels=None, **kwargs):
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
        problem = LinearLMEProblem.from_x_y(x, y=None, columns_labels=columns_labels)
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

        beta = self.coef_['beta']
        us = self.coef_['random_effects']

        assert problem.num_fixed_effects == beta.shape[0], \
            "Number of fixed effects is not the same to what it was in the train data."

        assert problem.num_random_effects == us[0].shape[0], \
            "Number of random effects is not the same to what it was in the train data."

        group_labels = self.coef_['group_labels']
        answers = []
        for i, (x, _, z, stds) in enumerate(problem):
            label = problem.group_labels[i]
            idx_of_this_label_in_train = np.where(group_labels == label)
            assert len(idx_of_this_label_in_train) <= 1, "Group labels of the classifier contain duplicates."
            if len(idx_of_this_label_in_train) == 1:
                idx_of_this_label_in_train = idx_of_this_label_in_train[0]
                y = x.dot(beta) + z.dot(us[idx_of_this_label_in_train][0])
            else:
                # If we have not seen this group (so we don't have inferred random effects for this)
                # then we make a prediction with "expected" (e.g. zero) random effects
                y = x.dot(beta)
            answers.append(y)
        return np.concatenate(answers)

    def score(self, x, y, sample_weight=None):
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

        sample_weight : array_like, Optional
            Weights of samples for calculating the R^2 statistics.

        Returns
        -------
        r2_score : float
            R^2 score

        """

        y_pred = self.predict(x)
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
            raise AssertionError("The model has not been fitted yet. Call .fit() first.")

    def muller2018ic(self):
        """
        Implements Information Criterion (IC) from (Muller, 2018)

        Returns
        -------
        Value of Mueller's IC
        """

        self.check_is_fitted()
        return self.oracle.muller2018ic(beta=self.coef_['beta'], gamma=self.coef_['gamma'])

    def vaida2005aic(self):
        """
        Calculates Akaike Information Criterion (AIC) from https://www.jstor.org/stable/2673485

        Returns
        -------
        Value for Vaida AIC
        """
        self.check_is_fitted()
        return self.oracle.vaida2005aic(beta=self.coef_['beta'], gamma=self.coef_['gamma'])

    def jones2010bic(self):
        """
        Implements Bayes Information Criterion (BIC) from (Jones, 2010)
        # https://www.researchgate.net/publication/51536734_Bayesian_information_criterion_for_longitudinal_and_clustered_data

        Returns
        -------
        Value of Jones's BIC
        """
        self.check_is_fitted()
        return self.oracle.jones2010bic(beta=self.coef_['beta'], gamma=self.coef_['gamma'])

    def flip_probabilities(self):
        """
        Estimates the probabilities of coordinates in beta to flip their signs
        under the normal posterior.

        Returns
        -------
        probabilities each coordinates of beta, as ndarray.
        """
        self.check_is_fitted()
        return self.oracle.flip_probabilities_beta(beta=self.coef_['beta'], gamma=self.coef_['gamma'])


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
        solver = PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                           fixed_step_len=5e-2 if not fixed_step_len else fixed_step_len)
        oracle = LinearLMEOracle(None, prior=prior)
        regularizer = DummyRegularizer()
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         initializer=initializer,
                         logger_keys=logger_keys)


class L0LmeModel(LMEModel):
    """
    Implements an L0-regularized Linear Mixed-Effect Model. It allows specifying the maximal number of
    non-zero fixed and random effects in your model
    """

    def __init__(self,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_solver: int = 1000,
                 stepping: str = "line-search",
                 nnz_tbeta: int = 1,
                 nnz_tgamma: int = 1,
                 participation_in_selection=None,
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
        participation_in_selection: ndarray of int
            0 if the feature should be affected by the regularizer, 0 otherwise
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
                           fixed_step_len=1 if not fixed_step_len else fixed_step_len)
        oracle = LinearLMEOracle(None, prior=prior)
        l0_regularizer = L0Regularizer(nnz_tbeta=nnz_tbeta,
                                       nnz_tgamma=nnz_tgamma,
                                       participation_in_selection=participation_in_selection,
                                       oracle=oracle)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=l0_regularizer)
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         initializer=initializer,
                         logger_keys=logger_keys)


class Sr3L0LmeModel(LMEModel):
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

    The problem should be provided as LinearLMEProblem.

    """

    def __init__(self,
                 tol_oracle: float = 1e-5,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_oracle: int = 1000,
                 max_iter_solver: int = 20,
                 stepping: str = "fixed",
                 lb: float = 40,
                 lg: float = 40,
                 nnz_tbeta: int = 1,
                 nnz_tgamma: int = 1,
                 participation_in_selection=None,
                 logger_keys: Set = ('converged',),
                 warm_start=True,
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
        participation_in_selection: ndarray of int
            0 if the feature should be affected by the regularizer, 0 otherwise
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        warm_start: bool
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
        solver = FakePGDSolver(update_prox_every=update_prox_every) if practical \
            else PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                           fixed_step_len=(
                               1 if max(lb, lg) == 0 else 1 / max(lb, lg)) if not fixed_step_len else fixed_step_len)
        oracle = LinearLMEOracleSR3(None, lb=lb, lg=lg, tol_inner=tol_oracle, n_iter_inner=max_iter_oracle,
                                    warm_start=warm_start, prior=prior)
        l0_regularizer = L0Regularizer(nnz_tbeta=nnz_tbeta,
                                       nnz_tgamma=nnz_tgamma,
                                       participation_in_selection=participation_in_selection,
                                       oracle=oracle)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=l0_regularizer)
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         initializer=initializer,
                         logger_keys=logger_keys)


class L1LmeModel(LMEModel):
    """
    Implements a LASSO-regularized Linear Mixed-Effect Model
    """

    def __init__(self,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_solver: int = 1000,
                 stepping: str = "line-search",
                 lam: float = 1,
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
        solver = PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                           fixed_step_len=1 / (lam + 1) if not fixed_step_len else fixed_step_len)
        oracle = LinearLMEOracle(None, prior=prior)
        l1_regularizer = L1Regularizer(lam=lam)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=l1_regularizer)
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         initializer=initializer,
                         logger_keys=logger_keys)


class SR3L1LmeModel(LMEModel):
    """
    Implements an SR3-relaxed LASSO-regularized Linear Mixed-Effect Model
    """

    def __init__(self,
                 tol_oracle: float = 1e-5,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_oracle: int = 1000,
                 max_iter_solver: int = 20,
                 stepping: str = "fixed",
                 lb: float = 40,
                 lg: float = 40,
                 lam: float = 1,
                 logger_keys: Set = ('converged',),
                 warm_start=True,
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
        lam: float
            strength of LASSO regularizer
        participation_in_selection: ndarray of int
            0 if the feature should be affected by the regularizer, 0 otherwise
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        warm_start: bool
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

        fixed_step_len = (1 if max(lb, lg) == 0 else 1 / max(lb, lg)) if not fixed_step_len else fixed_step_len
        solver = FakePGDSolver(fixed_step_len=fixed_step_len, update_prox_every=update_prox_every) if practical \
            else PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                           fixed_step_len=fixed_step_len)
        oracle = LinearLMEOracleSR3(None, lb=lb, lg=lg, tol_inner=tol_oracle, n_iter_inner=max_iter_oracle,
                                    warm_start=warm_start, prior=prior)
        l1_regularizer = L1Regularizer(lam=lam)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=l1_regularizer)
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         initializer=initializer,
                         logger_keys=logger_keys)


class CADLmeModel(LMEModel):
    """
    Implements a CAD-regularized Linear Mixed-Effects Model
    """

    def __init__(self,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_solver: int = 1000,
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
        solver = PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                           fixed_step_len=1 / (lam + 1) if not fixed_step_len else fixed_step_len)
        oracle = LinearLMEOracle(None, prior=prior)
        cad_regularizer = CADRegularizer(rho=rho, lam=lam)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=cad_regularizer)
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         initializer=initializer,
                         logger_keys=logger_keys)


class SR3CADLmeModel(LMEModel):
    """
    Implements a CAD-regularized SR3-relaxed Linear Mixed-Effect Model
    """

    def __init__(self,
                 tol_oracle: float = 1e-5,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_oracle: int = 1000,
                 max_iter_solver: int = 20,
                 stepping: str = "fixed",
                 lb: float = 40.0,
                 lg: float = 40.0,
                 rho: float = 1.0,
                 lam: float = 1.0,
                 logger_keys: Set = ('converged',),
                 warm_start=True,
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
        lam: float
            strength of CAD regularizer
        rho: float
            cut-off amplitude above which the coefficients are not penalized
        participation_in_selection: ndarray of int
            0 if the feature should be affected by the regularizer, 0 otherwise
        logger_keys: List[str]
            list of keys for the parameters that the logger should track
        warm_start: bool
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

        fixed_step_len = (1 if max(lb, lg) == 0 else 1 / max(lb, lg)) if not fixed_step_len else fixed_step_len
        solver = FakePGDSolver(update_prox_every=update_prox_every, fixed_step_len=fixed_step_len) if practical \
            else PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                           fixed_step_len=fixed_step_len)
        oracle = LinearLMEOracleSR3(None, lb=lb, lg=lg, tol_inner=tol_oracle, n_iter_inner=max_iter_oracle,
                                    warm_start=warm_start, prior=prior)
        cad_regularizer = CADRegularizer(rho=rho, lam=lam)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=cad_regularizer)
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         initializer=initializer,
                         logger_keys=logger_keys)


class SCADLmeModel(LMEModel):
    """
    Implements SCAD-regularized Linear Mixed-Effect Model
    """

    def __init__(self,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_solver: int = 1000,
                 stepping: str = "line-search",
                 rho: float = 3.7,  # as per recommendation from (Fan, Li, 2001), p. 1351
                 sigma: float = 1.6,  # same
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
        sigma: float, sigma > 1
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
        solver = PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                           fixed_step_len=1 / (lam + 1) if not fixed_step_len else fixed_step_len)
        oracle = LinearLMEOracle(None, prior=prior)
        scad_regularizer = SCADRegularizer(rho=rho, sigma=sigma, lam=lam)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=scad_regularizer)
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         initializer=initializer,
                         logger_keys=logger_keys)


class SR3SCADLmeModel(LMEModel):
    """
    Implements SCAD-regularized SR3-relaxed Linear Mixed-Effects Model
    """

    def __init__(self,
                 tol_oracle: float = 1e-5,
                 tol_solver: float = 1e-5,
                 initializer: str = "None",
                 max_iter_oracle: int = 1000,
                 max_iter_solver: int = 20,
                 stepping: str = "fixed",
                 lb: float = 40.0,
                 lg: float = 40.0,
                 rho: float = 3.7,  # as per recommendation from (Fan, Li, 2001), p. 1351
                 sigma: float = 1.6,  # same
                 lam: float = 1.0,
                 logger_keys: Set = ('converged',),
                 warm_start=True,
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
        warm_start: bool
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

        fixed_step_len = (1 if max(lb, lg) == 0 else 1 / max(lb, lg)) if not fixed_step_len else fixed_step_len
        solver = FakePGDSolver(update_prox_every=update_prox_every, fixed_step_len=fixed_step_len) if practical \
            else PGDSolver(tol=tol_solver, max_iter=max_iter_solver, stepping=stepping,
                           fixed_step_len=fixed_step_len)
        oracle = LinearLMEOracleSR3(None, lb=lb, lg=lg, tol_inner=tol_oracle, n_iter_inner=max_iter_oracle,
                                    warm_start=warm_start, prior=prior)
        scad_regularizer = SCADRegularizer(rho=rho, sigma=sigma, lam=lam)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=scad_regularizer)
        super().__init__(oracle=oracle,
                         solver=solver,
                         regularizer=regularizer,
                         initializer=initializer,
                         logger_keys=logger_keys)


def _check_input_consistency(problem, beta=None, gamma=None, tbeta=None, tgamma=None):
    """
    Checks the consistency of .fit() arguments

    Parameters
    ----------
    problem : LinearLMEProblem
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

    num_features = problem.num_fixed_effects
    num_random_effects = problem.num_random_effects
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
