from typing import Set

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from lib.problems import LinearLMEProblem
from lib.new_oracles import LinearLMEOracleRegularized
from lib.logger import Logger


class LinearLMESparseModel(BaseEstimator, RegressorMixin):
    """
    Solve Linear Mixed Effects problem with projected gradient descent method
    """

    def __init__(self, tol: float = 1e-4, tol_inner: float = 1e-4, solver: str = "pgd", n_iter=1000, n_iter_inner=20,
                 use_line_search=True, lb=1, lg=1,
                 nnz_tbeta=3, nnz_tgamma=3, logger_keys: Set = ('converged',)):
        """
        Initializes the model

        Parameters
        ----------
        tol : float
            Precision of the solution.

        solver : {'pgd'}
            Solver to use in computational routines:

            - 'pgd' Projected Gradient Descent

        """

        self.tol = tol
        self.tol_inner = tol_inner
        self.solver = solver
        self.n_iter = n_iter
        self.n_iter_inner = n_iter_inner
        self.use_line_search = use_line_search
        self.lb = lb
        self.lg = lg
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma
        self.logger_keys = logger_keys

    def fit(self, x, y, columns_labels=None, initial_parameters: dict = None, warm_start=False,
            initializer=None, **kwargs):
        """
        Fits a Linear Model with Linear Mixed-Effects to the given data

        Parameters
        ----------
        x : array-like
        y : array-like
        columns_labels
        initial_parameters:
            - gamma0 : array-like,
                If None then it defaults to an all-ones vector.
            - beta0 :
                If None then it defaults to an all-ones vector.
            - tbeta0 :
                If None then it defaults to an all-zeros vector.
            - tgamma0 :
                If None then it defaults to an all-zeros vector.
        warm_start :
            Use previous parameters
        initializer : {None, 'EM'}, Optional
            Which initializer to use
        kwargs

        Returns
        -------
        self :
            Fitted regression model.
        """

        problem = LinearLMEProblem.from_x_y(x, y, columns_labels, **kwargs)
        beta0 = initial_parameters.get("beta", None)
        gamma0 = initial_parameters.get("gamma", None)
        tbeta0 = initial_parameters.get("tbeta", None)
        tgamma0 = initial_parameters.get("tgamma", None)
        _check_input_consistency(problem, beta0, gamma0, tbeta0, tgamma0)

        oracle = LinearLMEOracleRegularized(problem,
                                            lb=self.lb,
                                            lg=self.lg,
                                            nnz_tbeta=self.nnz_tbeta,
                                            nnz_tgamma=self.nnz_tgamma
                                            )
        num_features = problem.num_features
        num_random_effects = problem.num_random_effects

        if hasattr(self, 'coef_') and warm_start:
            beta = self.coef_["beta"]
            gamma = self.coef_["gamma"]
            tbeta = self.coef_["tbeta"]
            tgamma = self.coef_["tgamma"]
        else:
            if beta0 is not None:
                beta = beta0
            else:
                beta = np.ones(num_features)

            if gamma0 is not None:
                gamma = gamma0
            else:
                gamma = np.ones(num_random_effects)

            if tbeta0 is not None:
                tbeta = tbeta0
            else:
                tbeta = np.zeros(num_features)

            if tgamma0 is not None:
                tgamma = tgamma0
            else:
                tgamma = np.zeros(num_random_effects)

        if initializer == "EM":
            beta = oracle.optimal_beta(gamma)
            us = oracle.optimal_random_effects(beta, gamma)
            gamma = np.sum(us ** 2, axis=0) / oracle.problem.num_studies

        loss = oracle.loss(beta, gamma, tbeta, tgamma)
        gradient_gamma = oracle.gradient_gamma(beta, gamma, tgamma)
        projected_gradient = gradient_gamma.copy()
        projected_gradient[gamma == 0] = 0
        self.logger_ = Logger(self.logger_keys)

        prev_tbeta = np.infty
        prev_tgamma = np.infty

        iteration = 0
        while (np.linalg.norm(tbeta - prev_tbeta) > self.tol
               and np.linalg.norm(tgamma - prev_tgamma) > self.tol
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
                while (np.linalg.norm(projected_gradient) > self.tol_inner
                       and inner_iteration < self.n_iter_inner):
                    beta = oracle.optimal_beta(gamma, tbeta)
                    gradient_gamma = oracle.gradient_gamma(beta, gamma, tgamma)
                    projected_gradient = gradient_gamma.copy()
                    projected_gradient[gamma == 0] = 0
                    direction = -projected_gradient
                    if self.use_line_search:
                        # line search method
                        step_len = 1
                        current_loss = oracle.loss(beta, gamma, tbeta, tgamma)
                        while oracle.loss(beta, gamma + step_len * direction, tbeta, tgamma) >= current_loss:
                            step_len *= 0.5
                            if step_len <= 1e-15:
                                break
                    else:
                        # fixed step size
                        step_len = 1 / iteration
                    gamma = gamma + step_len * direction

                    # projection
                    gamma = np.clip(gamma, 0, None)

                    gradient_gamma = oracle.gradient_gamma(beta, gamma, tgamma)
                    projected_gradient = gradient_gamma.copy()
                    projected_gradient[gamma == 0] = 0
                    inner_iteration += 1

                prev_tbeta = tbeta
                prev_tgamma = tgamma
                tbeta = oracle.optimal_tbeta(beta)
                tgamma = oracle.optimal_tgamma(tbeta, gamma)

            if len(self.logger_keys) > 0:
                self.logger_.log(**locals())

        us = oracle.optimal_random_effects(beta, gamma)
        per_cluster_coefficients = np.zeros(problem.num_studies, len(problem.coefficients_to_columns_mapping))
        for i, u in enumerate(us):
            for j, (idx_beta, idx_u) in enumerate(problem.coefficients_to_columns_mapping):
                if idx_beta is not None:
                    per_cluster_coefficients[i, j] += beta[idx_beta]
                if idx_u is not None:
                    per_cluster_coefficients[i, j] += u[idx_u]

        self.logger_.add('converged', 1)
        self.logger_.add('iterations', iteration)

        self.coef_ = {
            "beta": beta,
            "gamma": gamma,
            "tbeta": tbeta,
            "tgamma": tgamma,
            "random_effects": us,
            "group_labels": np.copy(problem.group_labels),
            "per_cluster_coefficients": per_cluster_coefficients
        }

        return self

    def predict(self, data):
        check_is_fitted(self, 'coef_')
        problem = LinearLMEProblem.from_x_y(data, y=None)
        beta = self.coef_['beta']
        us = self.coef_['random_effects']
        group_labels = self.coef_['group_labels']
        answers = []
        for i, (x, _, z, stds) in enumerate(problem):
            label = problem.group_labels[i]
            idx_of_this_label_in_train = np.where(group_labels == label)
            assert len(idx_of_this_label_in_train) <= 1, "Group labels of the classifier contain duplicates."
            if len(idx_of_this_label_in_train) == 1:
                idx_of_this_label_in_train = idx_of_this_label_in_train[0]
                y = x.dot(beta) + z.dot(us[idx_of_this_label_in_train])
            else:
                # If we have not seen this group (so we don't have inferred random effects for this)
                # then we make a prediction with "expected" (e.g. zero) random effects
                y = x.dot(beta)
            answers.append(y)
        return answers


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
        Vector of the sparse set of random effecta (for regularized models)

    Returns
    -------
        output : None
            None if all the checks are passed, otherwise raises an exception

    """

    num_features = problem.num_features
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
