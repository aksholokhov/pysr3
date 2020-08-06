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

from typing import Set

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracleRegularized, LinearLMEOracleW
from skmixed.logger import Logger
from skmixed.helpers import get_per_group_coefficients


class LinearLMESparseModel(BaseEstimator, RegressorMixin):
    """
    Solve regularized sparse Linear Mixed Effects problem with projected gradient descent method.

    The log-likelihood minimization problem which this model solves is::

        min ℋ(β, 𝛄, tβ, t𝛄) w.r.t. all four arguments (β, 𝛄, tβ, t𝛄)
        s.t. nnz(tbeta) <= nnz_tbeta and nnz(gamma) <= nnz_tgamma where

        ℋ(β, 𝛄, tβ, t𝛄) := ℒ(β, 𝛄) + lb/2*||β - tβ||^2 + lg/2*||𝛄 - t𝛄||^2

        ℒ(β, 𝛄) = ∑(yi - Xi*β)ᵀΩi^{-1}(yi - Xi*β) + ln(det(Ωi))

        Ωi = Zi*diag(𝛄)Ziᵀ + diag(obs_stds)

    The original statistical model which this loss is based on is::

        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        u_i ~ 𝒩(0, diag(𝛄)),

        𝛄 ~ 𝒩(t𝛄, 1/lg)

        β ~ 𝒩(tβ, 1/lb)

        𝜺_i ~ 𝒩(0, diag(obs_std)

    See my paper for more details.
    """

    def __init__(self,
                 tol: float = 1e-4,
                 tol_inner: float = 1e-4,
                 tol_outer: float = 1e-2,
                 solver: str = "pgd",
                 initializer=None,
                 n_iter: int = 1000,
                 n_iter_inner: int = 20,
                 n_iter_outer: int = 1,
                 use_line_search: bool = True,
                 lb: float = 1,
                 lg: float = 1,
                 regularization_type: str = "l2",
                 nnz_tbeta: int = 3,
                 nnz_tgamma: int = 3,
                 logger_keys: Set = ('converged',)):
        """
        init: initializes the model.

        Parameters
        ----------
        tol : float
            Tolerance for stopping criterion: ||tβ_{k+1} - tβ_k|| <= tol and ||t𝛄_{k+1} - t𝛄_k|| <= tol.

        tol_inner : float
            Tolerance for inner optimization subroutine (min ℋ w.r.t. 𝛄) stopping criterion:
            ||projected ∇ℋ|| <= tol_inner

        solver : {'pgd'} Solver to use in computational routines:

                - 'pgd' : Projected Gradient Descent

        initializer : {None, 'EM'}, Optional
            Whether to use an initializer before starting the main optimization routine:

                - None : Does not do any special initialization, starts with the given initial point.
                - 'EM' : Performs one step of a naive EM-algorithm in order to improve the initial point.

        n_iter : int
            Number of iterations for the outer optimization cycle.

        n_iter_inner : int
            Number of iterations for the inner optimization cycle.

        use_line_search : bool, default = True
            Whether to use line search when optimizing w.r.t. 𝛄. If true, it starts from step_len = 1 and cuts it
            in half until the descent criterion is met. If false, it uses a fixed step size of 1/iteration_number.

        lb : float
            Regularization coefficient for the tβ-related term, see the loss-function description.

        lg : float
            Regularization coefficient for the t𝛄-related term, see the loss-function description.

        regularization_type : str, one of {"l2", "loss-weighted"}, default = "l2"
            Type of norm in the regularization terms. Options are:

                - 'l2' : Euclidean norm.
                - 'loss-weighted' : A weighted norm designed so to drop the least important coefficients.

        nnz_tbeta : int,
            How many non-zero coefficients are allowed in tβ.

        nnz_tgamma : int,
            How many non-zero coefficients are allowed in t𝛄.
        """

        self.tol = tol
        self.tol_inner = tol_inner
        self.tol_outer = tol_outer
        self.solver = solver
        self.initializer = initializer
        self.n_iter = n_iter
        self.n_iter_inner = n_iter_inner
        self.n_iter_outer = n_iter_outer
        self.use_line_search = use_line_search
        self.lb = lb
        self.lg = lg
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma
        self.logger_keys = logger_keys
        self.regularization_type = regularization_type

    def fit_problem(self,
                    problem: LinearLMEProblem,
                    columns_labels: np.ndarray = None,
                    initial_parameters: dict = None,
                    warm_start=False,
                    random_intercept=True,
                    **kwargs):
        """
        Fits a Linear Model with Linear Mixed-Effects to the given data.

        Parameters
        ----------
        problem : LinearLMEProblem
            The problem to fit the model into

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
                    | Initial estimate of random effects covariances. If None then it defaults to an all-ones vector.
                -   | 'tbeta0' : np.ndarray, shape = [n],
                    | Initial estimate of sparse fixed effects. If None then it defaults to an all-zeros vector.
                -   | 'tgamma0' : np.ndarray, shape = [k],
                    | Initial estimate of sparse random covariances. If None then it defaults to an all-zeros vector.

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

        if initial_parameters is None:
            initial_parameters = {}
        beta0 = initial_parameters.get("beta", None)
        gamma0 = initial_parameters.get("gamma", None)
        tbeta0 = initial_parameters.get("tbeta", None)
        tgamma0 = initial_parameters.get("tgamma", None)
        _check_input_consistency(problem, beta0, gamma0, tbeta0, tgamma0)

        if self.regularization_type == "l2":
            oracle = LinearLMEOracleRegularized(problem,
                                                lb=self.lb,
                                                lg=self.lg,
                                                nnz_tbeta=self.nnz_tbeta,
                                                nnz_tgamma=self.nnz_tgamma
                                                )
        elif self.regularization_type == "loss-weighted":
            oracle = LinearLMEOracleW(problem,
                                      lb=self.lb,
                                      lg=self.lg,
                                      nnz_tbeta=self.nnz_tbeta,
                                      nnz_tgamma=self.nnz_tgamma
                                      )
        else:
            raise ValueError("regularization_type is not understood.")

        num_fixed_effects = problem.num_fixed_effects
        num_random_effects = problem.num_random_effects
        assert num_fixed_effects >= self.nnz_tbeta
        assert num_random_effects >= self.nnz_tgamma
        # old_oracle = OldOracle(problem, lb=self.lb, lg=self.lg, k=self.nnz_tbeta, j=self.nnz_tgamma)

        if warm_start:
            check_is_fitted(self, 'coef_')
            beta = self.coef_["beta"]
            gamma = self.coef_["gamma"]
            tbeta = self.coef_["tbeta"]
            tgamma = self.coef_["tgamma"]

        else:
            if beta0 is not None:
                beta = beta0
            else:
                beta = np.ones(num_fixed_effects)

            if gamma0 is not None:
                gamma = gamma0
            else:
                gamma = np.ones(num_random_effects)

            if tbeta0 is not None:
                tbeta = tbeta0
            else:
                tbeta = np.zeros(num_fixed_effects)

            if tgamma0 is not None:
                tgamma = tgamma0
            else:
                tgamma = np.zeros(num_random_effects)

        if self.initializer == "EM":
            beta = oracle.optimal_beta(gamma, tbeta, beta=beta)
            us = oracle.optimal_random_effects(beta, gamma)
            gamma = np.sum(us ** 2, axis=0) / oracle.problem.num_groups
            # tbeta = oracle.optimal_tbeta(beta)
            # tgamma = oracle.optimal_tgamma(tbeta, gamma)

        def projected_direction(current_gamma, current_direction):
            proj_direction = current_direction.copy()
            for j, _ in enumerate(current_gamma):
                if current_gamma[j] <= 1e-15 and current_direction[j] <= 0:
                    proj_direction[j] = 0
            return proj_direction

        loss = oracle.loss(beta, gamma, tbeta, tgamma)
        self.logger_ = Logger(self.logger_keys)

        outer_iteration = 0

        while (outer_iteration < self.n_iter_outer
               and (np.linalg.norm(beta - tbeta) > self.tol_outer
                    or np.linalg.norm(gamma - tgamma) > self.tol_outer)):

            prev_tbeta = np.infty
            prev_tgamma = np.infty
            prev_beta = np.infty
            prev_gamma = np.infty

            iteration = 0
            while ((np.linalg.norm(tbeta - prev_tbeta) > self.tol
                    or np.linalg.norm(tgamma - prev_tgamma) > self.tol
                    or np.linalg.norm(beta - prev_beta) > self.tol
                    or np.linalg.norm(gamma - prev_gamma) > self.tol)
                   and iteration < self.n_iter):

                if iteration >= self.n_iter:
                    us = oracle.optimal_random_effects(beta, gamma)
                    if len(self.logger_keys) > 0:
                        self.logger_.log(**locals())
                    self.coef_ = {"beta": beta,
                                  "gamma": gamma,
                                  "tbeta": tbeta,
                                  "tgamma": tgamma,
                                  "random_effects": us
                                  }
                    self.logger_.add("converged", 0)
                    return self

                if self.solver == 'pgd':
                    inner_iteration = 0

                    prev_beta = beta
                    prev_gamma = gamma
                    prev_tbeta = tbeta
                    prev_tgamma = tgamma

                    beta = oracle.optimal_beta(gamma, tbeta, beta=beta)
                    gradient_gamma = oracle.gradient_gamma(beta, gamma, tgamma)
                    direction = projected_direction(gamma, -gradient_gamma)
                    while (np.linalg.norm(direction) > self.tol_inner
                           and inner_iteration < self.n_iter_inner):
                        # gradient_gamma = oracle.gradient_gamma(beta, gamma, tgamma)
                        # projecting the gradient to the set of constraints
                        # direction = projected_direction(gamma, -gradient_gamma)
                        if self.use_line_search:
                            # line search method
                            step_len = 0.1
                            for i, _ in enumerate(gamma):
                                if direction[i] < 0:
                                    step_len = min(-gamma[i] / direction[i], step_len)

                            current_loss = oracle.loss(beta, gamma, tbeta, tgamma)

                            while (oracle.loss(beta, gamma + step_len * direction, tbeta, tgamma)
                                   >= (1 - np.sign(current_loss) * 1e-5) * current_loss):
                                step_len *= 0.5
                                if step_len <= 1e-15:
                                    break
                        else:
                            # fixed step size
                            step_len = 1 / iteration
                        if step_len <= 1e-15:
                            break
                        gamma = gamma + step_len * direction
                        gradient_gamma = oracle.gradient_gamma(beta, gamma, tgamma)
                        direction = projected_direction(gamma, -gradient_gamma)
                        inner_iteration += 1

                    tbeta = oracle.optimal_tbeta(beta=beta, gamma=gamma)
                    tgamma = oracle.optimal_tgamma(tbeta, gamma, beta=beta)
                    iteration += 1

                loss = oracle.loss(beta, gamma, tbeta, tgamma)
                if len(self.logger_keys) > 0:
                    self.logger_.log(locals())
            # gradient_at_tgamma = oracle.gradient_gamma(tbeta, tgamma, tgamma)
            # direction_at_sparse_point = projected_direction(tgamma, -gradient_at_tgamma)
            outer_iteration += 1
            oracle.lb = 2 * (1 + oracle.lb)
            oracle.lg = 2 * (1 + oracle.lg)

        us = oracle.optimal_random_effects(beta, gamma)
        sparse_us = oracle.optimal_random_effects(tbeta, tgamma)

        per_group_coefficients = get_per_group_coefficients(beta, us, labels=problem.column_labels)
        sparse_per_group_coefficients = get_per_group_coefficients(tbeta, sparse_us, labels=problem.column_labels)

        self.logger_.add('converged', 1)
        # self.logger_.add('iterations', iteration) # TODO: fix inner + outer iterations counter
        for key in self.logger_keys:
            if key.startswith("IC_"):
                self.logger_.add(key, oracle.get_ic(key, beta=beta, gamma=gamma, tbeta=tbeta, tgamma=tgamma))

        self.coef_ = {
            "beta": beta,
            "gamma": gamma,
            "tbeta": tbeta,
            "tgamma": tgamma,
            "random_effects": us,
            "sparse_random_effects": sparse_us,
            "group_labels": np.copy(problem.group_labels),
            "per_group_coefficients": per_group_coefficients,
            "sparse_per_group_coefficients": sparse_per_group_coefficients,
        }

        return self

    def predict(self, x, columns_labels=None, use_sparse_coefficients=False):
        """
        Makes a prediction if .fit(X, y) was called before and throws an error otherwise.

        Parameters
        ----------
        x : np.ndarray
            Data matrix. Should have the same format as the data which was used for fitting the model:
            the number of columns and the columns' labels should be the same. It may contain new groups, in which case
            the prediction will be formed using the fixed effects only.

        use_sparse_coefficients : bool, default is False
            If true then uses sparse coefficients, tbeta and tgamma, for making a prediction, otherwise uses
            beta and gamma.

        Returns
        -------
        y : np.ndarray
            Models predictions.
        """
        check_is_fitted(self, 'coef_')
        problem = LinearLMEProblem.from_x_y(x, y=None, columns_labels=columns_labels)
        return self.predict_problem(problem, use_sparse_coefficients=True)

    def predict_problem(self, problem, use_sparse_coefficients=False):
        """
        Makes a prediction if .fit(X, y) was called before and throws an error otherwise.

        Parameters
        ----------
        x : np.ndarray
            Data matrix. Should have the same format as the data which was used for fitting the model:
            the number of columns and the columns' labels should be the same. It may contain new groups, in which case
            the prediction will be formed using the fixed effects only.

        use_sparse_coefficients : bool, default is False
            If true then uses sparse coefficients, tbeta and tgamma, for making a prediction, otherwise uses
            beta and gamma.

        Returns
        -------
        y : np.ndarray
            Models predictions.
        """
        check_is_fitted(self, 'coef_')

        if use_sparse_coefficients:
            beta = self.coef_['tbeta']
            us = self.coef_['sparse_random_effects']
        else:
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
                    | Initial estimate of random effects covariances. If None then it defaults to an all-ones vector.
                -   | 'tbeta0' : np.ndarray, shape = [n],
                    | Initial estimate of sparse fixed effects. If None then it defaults to an all-zeros vector.
                -   | 'tgamma0' : np.ndarray, shape = [k],
                    | Initial estimate of sparse random covariances. If None then it defaults to an all-zeros vector.

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
        return self.fit_problem(problem,
                                columns_labels=columns_labels,
                                initial_parameters=initial_parameters,
                                warm_start=warm_start,
                                random_intercept=random_intercept)


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
