# This code implements a variety of sklearn-compatible linear mixed-effects models.
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

"""Linear Mixed-Effects Models (simple, relaxed, and regularized)"""

import warnings
from typing import Set, Optional, Tuple, List

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, check_X_y, check_array
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils.validation import check_is_fitted

from pysr3.lme.oracles import LinearLMEOracle, LinearLMEOracleSR3
from pysr3.lme.problems import LMEProblem, FIXED, RANDOM, FIXED_RANDOM
from pysr3.lme.problems import get_per_group_coefficients
from pysr3.logger import Logger
from pysr3.regularizers import Regularizer, L0RegularizerLME, L1Regularizer, CADRegularizer, SCADRegularizer, \
    DummyRegularizer, PositiveQuadrantRegularizer, ElasticRegularizer
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
        initializer: str
            "EM" or "None"
        logger_keys: Optional[Set[str]]
            list of values for the logger to track.
        """
        self.logger_keys = logger_keys
        self.initializer = initializer

    def instantiate(self) -> Tuple[Optional[LinearLMEOracle], Optional[Regularizer], Optional[PGDSolver]]:
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver.
        """
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
            Data matrix. Rows correspond to objects, columns correspond to features, group labels, and variances.

        y : np.ndarray
            Answers, real-valued array.

        columns_labels :  List[str]
            List of column labels. There shall be only one column of group labels and answers STDs.

                - "fixed" : fixed effect
                - "random" : random effect
                - "fixed+random" : both fixed and random,
                - "group" : groups labels
                - "variance" : answers standard deviations
                -   | "intercept" : intercept column (fixed or random intercept is controlled by "fit_fixed_intercept"
                    | and "fit_random_intercept" respectively.

        initial_parameters : Dict[np.ndarray]
            Dict with possible fields:

                -   | 'beta' : np.ndarray, shape = [p],
                    | Initial estimate of fixed effects. If None then it defaults to an all-ones vector.
                -   | 'gamma' : np.ndarray, shape = [q],
                    | Initial estimate of random effects covariances.
                    | If None then it defaults to an all-ones vector.

        warm_start : bool, default is False
            Whether to use previous parameters as initial ones. Overrides initial_parameters if given.
            Throws NotFittedError if set to True when not fitted.

        fit_fixed_intercept : bool, default = False
            Whether to add the intercept to the model

        fit_random_intercept : bool, default = False
            Whether treat the intercept as a random effect.

        fe_regularization_weights: ndarray[int], 0 or 1
            Vector of length of the number of features where 0 means do not apply regularizer to the
            coefficients of the corresponding fixed features and 1 means apply as usual.

        re_regularization_weights: ndarray[int], 0 or 1
            Vector of length of the number of features where 0 means do not apply regularizer to the
            coefficients of the corresponding fixed features and 1 means apply as usual.

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

                -   | 'beta' : np.ndarray, shape = [p],
                    | Initial estimate of fixed effects. If None then it defaults to an all-ones vector.
                -   | 'gamma' : np.ndarray, shape = [q],
                    | Initial estimate of random effects covariances. If None then it defaults to an all-ones vector.

        warm_start : bool, default is False
            Whether to use previous parameters as initial ones. Overrides initial_parameters if given.
            Throws NotFittedError if set to True when not fitted.

        fe_regularization_weights: ndarray[int], 0 or 1
            Vector of length of the number of features where 0 means do not apply regularizer to the
            coefficients of the corresponding fixed features and 1 means apply as usual.

        re_regularization_weights: ndarray[int], 0 or 1
            Vector of length of the number of features where 0 means do not apply regularizer to the
            coefficients of the corresponding fixed features and 1 means apply as usual.


        kwargs :
            Not used currently, left here for passing debugging parameters.

        Returns
        -------
            self
        """
        oracle, regularizer, solver = self.instantiate()

        oracle.instantiate(problem)
        if fe_regularization_weights is None:
            if problem.fe_regularization_weights is None:
                fe_regularization_weights = np.ones(problem.num_fixed_features)
                if problem.intercept_label == FIXED or problem.intercept_label == FIXED_RANDOM:
                    fe_regularization_weights[0] = 0
            else:
                fe_regularization_weights = problem.fe_regularization_weights
        if re_regularization_weights is None:
            if problem.re_regularization_weights is None:
                re_regularization_weights = np.ones(problem.num_random_features)
                if problem.intercept_label == RANDOM or problem.intercept_label == FIXED_RANDOM:
                    re_regularization_weights[0] = 0
            else:
                re_regularization_weights = problem.re_regularization_weights
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

        per_group_coefficients = get_per_group_coefficients(beta, us, labels=[problem.intercept_label]
                                                                             + problem.column_labels)

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

        columns_labels :  List[str]
            List of column labels. There shall be only one column of group labels and answers STDs.

                - "fixed" : fixed effect
                - "random" : random effect
                - "fixed+random" : both fixed and random,
                - "group" : groups labels
                - "variance" : answers standard deviations
                -   | "intercept" : intercept column (fixed or random intercept is controlled by "fit_fixed_intercept"
                    | and "fit_random_intercept" respectively.

        fit_fixed_intercept: bool, default = False
            Whether to add an intercept as a fixed feature

        fit_random_intercept: bool, default = False
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
        for i, (x, _, z, _) in enumerate(problem):
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

        columns_labels :  List[str]
            List of column labels. There shall be only one column of group labels and answers STDs.

                - "fixed" : fixed effect
                - "random" : random effect
                - "fixed+random" : both fixed and random,
                - "group" : groups labels
                - "variance" : answers standard deviations
                -   | "intercept" : intercept column (fixed or random intercept is controlled by "fit_fixed_intercept"
                    | and "fit_random_intercept" respectively.

        fit_fixed_intercept: bool, default = False
            Whether to add an intercept as a fixed feature

        fit_random_intercept: bool, default = False
            Whether to add an intercept as a random feature.

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
                 elastic_eps: float = 1e-4,
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
        self.elastic_eps = elastic_eps

    def instantiate(self):
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver

        Returns
        -------
        Tuple of [Oracle, Regularizer, Solver] that correspond to this model
        """
        oracle = LinearLMEOracle(None, prior=self.prior)
        dummy_regularizer = DummyRegularizer()
        elastic_regularizer = ElasticRegularizer(other_regularizer=dummy_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
        fixed_step_len = 5e-2 if not self.fixed_step_len else self.fixed_step_len
        solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                           fixed_step_len=fixed_step_len)

        return oracle, regularizer, solver

    def get_information_criterion(self, x, y, columns_labels=None, ic="muller_ic"):
        """
        x : np.ndarray
            Data matrix. Rows correspond to objects, columns correspond to features, group labels, and variances.

        y : np.ndarray
            Answers, real-valued array.

        columns_labels :  List[str]
            List of column labels. There shall be only one column of group labels and answers STDs.

                - "fixed" : fixed effect
                - "random" : random effect
                - "fixed+random" : both fixed and random,
                - "group" : groups labels
                - "variance" : answers standard deviations
                -   | "intercept" : intercept column (fixed or random intercept is controlled by "fit_fixed_intercept"
                    | and "fit_random_intercept" respectively.

        ic : str
            Information criterion. Can be one of the following

                - "muller_ic": IC from (Hui, Muller, 2016)
                - "vaida_aic": AIC from (Vaida, 2005)
                - "jones_bic": BIC from (Jones, 2010)

        Returns
        -------
            value of the requested IC
        """
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
                 elastic_eps: float = 1e-4,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 update_prox_every=1,
                 fixed_step_len=None,
                 take_only_positive_part=True,
                 take_expected_value=False,
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
        self.elastic_eps = elastic_eps
        self.practical = practical
        self.take_only_positive_part = take_only_positive_part
        self.take_expected_value = take_expected_value
        self.update_prox_every = update_prox_every

    def instantiate(self):
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver

        Returns
        -------
        Tuple of [Oracle, Regularizer, Solver] that correspond to this model
        """
        if not self.fixed_step_len:
            fixed_step_len = 1 if self.ell == 0 else 1 / self.ell
        else:
            fixed_step_len = self.fixed_step_len
        if self.practical:
            solver = FakePGDSolver(update_prox_every=self.update_prox_every)
        else:
            solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                               fixed_step_len=fixed_step_len)
        oracle = LinearLMEOracleSR3(None, lb=self.ell, lg=self.ell,
                                    tol_inner=self.tol_oracle,
                                    n_iter_inner=self.max_iter_oracle,
                                    warm_start=self.warm_start_oracle,
                                    take_only_positive_part=self.take_only_positive_part,
                                    take_expected_value=self.take_expected_value,
                                    prior=self.prior)
        dummy_regularizer = DummyRegularizer()
        elastic_regularizer = ElasticRegularizer(other_regularizer=dummy_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
        return oracle, regularizer, solver

    def get_information_criterion(self, x, y, columns_labels=None, ic="muller_ic"):
        """
        x : np.ndarray
            Data matrix. Rows correspond to objects, columns correspond to features, group labels, and variances.

        y : np.ndarray
            Answers, real-valued array.

        columns_labels :  List[str]
            List of column labels. There shall be only one column of group labels and answers STDs.

                - "fixed" : fixed effect
                - "random" : random effect
                - "fixed+random" : both fixed and random,
                - "group" : groups labels
                - "variance" : answers standard deviations
                -   | "intercept" : intercept column (fixed or random intercept is controlled by "fit_fixed_intercept"
                    | and "fit_random_intercept" respectively.

        ic : str
            Information criterion. Can be one of the following

                - "muller_ic": IC from (Hui, Muller, 2016)
                - "vaida_aic": AIC from (Vaida, 2005)
                - "jones_bic": BIC from (Jones, 2010)

        Returns
        -------
            value of the requested IC
        """
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
                 elastic_eps: float = 1e-4,
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
                         elastic_eps=elastic_eps,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         prior=prior)
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma

    def instantiate(self):
        oracle, regularizer, solver = super().instantiate()
        l0_regularizer = L0RegularizerLME(nnz_tbeta=self.nnz_tbeta,
                                          nnz_tgamma=self.nnz_tgamma,
                                          oracle=oracle)
        elastic_regularizer = ElasticRegularizer(other_regularizer=l0_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
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
                 elastic_eps: float = 1e-4,
                 nnz_tbeta: int = 1,
                 nnz_tgamma: int = 1,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 take_only_positive_part=True,
                 take_expected_value=False,
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
                         take_only_positive_part=take_only_positive_part,
                         take_expected_value=take_expected_value,
                         update_prox_every=update_prox_every,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         ell=ell,
                         elastic_eps=elastic_eps,
                         prior=prior)
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma

    def instantiate(self):
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver

        Returns
        -------
        Tuple of [Oracle, Regularizer, Solver] that correspond to this model
        """
        oracle, regularizer, solver = super().instantiate()
        l0_regularizer = L0RegularizerLME(nnz_tbeta=self.nnz_tbeta,
                                          nnz_tgamma=self.nnz_tgamma,
                                          oracle=oracle)
        elastic_regularizer = ElasticRegularizer(other_regularizer=l0_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
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
                 elastic_eps: float = 1e-4,
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
                         elastic_eps=elastic_eps,
                         prior=prior)
        self.lam = lam

    def instantiate(self):
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver

        Returns
        -------
        Tuple of [Oracle, Regularizer, Solver] that correspond to this model
        """
        oracle, regularizer, solver = super().instantiate()
        fixed_step_len = 1 / (self.lam + 1) if not self.fixed_step_len else self.fixed_step_len
        solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                           fixed_step_len=fixed_step_len)
        l1_regularizer = L1Regularizer(lam=self.lam)
        elastic_regularizer = ElasticRegularizer(other_regularizer=l1_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
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
                 elastic_eps: float = 1e-4,
                 lam: float = 1,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 take_only_positive_part=True,
                 take_expected_value=False,
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
                         take_only_positive_part=take_only_positive_part,
                         take_expected_value=take_expected_value,
                         update_prox_every=update_prox_every,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         ell=ell,
                         elastic_eps=elastic_eps,
                         prior=prior)
        self.lam = lam

    def instantiate(self):
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver

        Returns
        -------
        Tuple of [Oracle, Regularizer, Solver] that correspond to this model
        """
        oracle, regularizer, solver = super().instantiate()
        l1_regularizer = L1Regularizer(lam=self.lam)
        elastic_regularizer = ElasticRegularizer(other_regularizer=l1_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
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
                 elastic_eps: float = 1e-4,
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
                         elastic_eps=elastic_eps,
                         prior=prior)
        self.lam = lam
        self.rho = rho

    def instantiate(self):
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver

        Returns
        -------
        Tuple of [Oracle, Regularizer, Solver] that correspond to this model
        """
        oracle, regularizer, solver = super().instantiate()
        fixed_step_len = 1 / (self.lam + 1) if not self.fixed_step_len else self.fixed_step_len
        solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                           fixed_step_len=fixed_step_len)
        cad_regularizer = CADRegularizer(lam=self.lam, rho=self.rho)
        elastic_regularizer = ElasticRegularizer(other_regularizer=cad_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
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
                 elastic_eps: float = 1e-4,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 take_only_positive_part=True,
                 take_expected_value=False,
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
                         take_only_positive_part=take_only_positive_part,
                         take_expected_value=take_expected_value,
                         update_prox_every=update_prox_every,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         ell=ell,
                         elastic_eps=elastic_eps,
                         prior=prior)
        self.lam = lam
        self.rho = rho

    def instantiate(self):
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver

        Returns
        -------
        Tuple of [Oracle, Regularizer, Solver] that correspond to this model
        """
        oracle, regularizer, solver = super().instantiate()
        cad_regularizer = CADRegularizer(lam=self.lam, rho=self.rho)
        elastic_regularizer = ElasticRegularizer(other_regularizer=cad_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
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
                 elastic_eps: float = 1e-4,
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
                         elastic_eps=elastic_eps,
                         prior=prior)
        self.lam = lam
        self.rho = rho
        self.sigma = sigma

    def instantiate(self):
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver

        Returns
        -------
        Tuple of [Oracle, Regularizer, Solver] that correspond to this model
        """
        oracle, regularizer, solver = super().instantiate()
        fixed_step_len = 1 / (self.lam + 1) if not self.fixed_step_len else self.fixed_step_len
        solver = PGDSolver(tol=self.tol_solver, max_iter=self.max_iter_solver, stepping=self.stepping,
                           fixed_step_len=fixed_step_len)
        scad_regularizer = SCADRegularizer(lam=self.lam, rho=self.rho, sigma=self.sigma)
        elastic_regularizer = ElasticRegularizer(other_regularizer=scad_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
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
                 elastic_eps: float = 1e-4,
                 logger_keys: Set = ('converged',),
                 warm_start_oracle=True,
                 practical=False,
                 take_only_positive_part=True,
                 take_expected_value=False,
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
                         take_only_positive_part=take_only_positive_part,
                         take_expected_value=take_expected_value,
                         update_prox_every=update_prox_every,
                         logger_keys=logger_keys,
                         fixed_step_len=fixed_step_len,
                         ell=ell,
                         elastic_eps=elastic_eps,
                         prior=prior)
        self.lam = lam
        self.rho = rho
        self.sigma = sigma

    def instantiate(self):
        """
        Instantiates the model: creates all internal entities such as oracle, regularizer, and solver

        Returns
        -------
        Tuple of [Oracle, Regularizer, Solver] that correspond to this model
        """
        oracle, regularizer, solver = super().instantiate()
        scad_regularizer = SCADRegularizer(lam=self.lam, rho=self.rho, sigma=self.sigma)
        elastic_regularizer = ElasticRegularizer(other_regularizer=scad_regularizer, eps=self.elastic_eps)
        regularizer = PositiveQuadrantRegularizer(other_regularizer=elastic_regularizer)
        return oracle, regularizer, solver
