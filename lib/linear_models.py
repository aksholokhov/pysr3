from typing import Set

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from lib.problems import LinearLMEProblem
from lib.new_oracles import LinearLMEOracleRegularized
from lib.logger import Logger

from lib.legacy.oracles import LinearLMEOracleRegularized as OldOracle

class LinearLMESparseModel(BaseEstimator, RegressorMixin):
    """
    Solve regularized sparse Linear Mixed Effects problem with projected gradient descent method:

    min w.r.t. β, 𝛄, tβ, t𝛄 the loss function

     ℒr(β, 𝛄, tβ, t𝛄) := ℒ(β, 𝛄) + lb/2*||β - tβ||^2 + lg/2*||𝛄 - t𝛄||^2
    """

    def __init__(self,
                 tol: float = 1e-4,
                 tol_inner: float = 1e-4,
                 solver: str = "pgd",
                 initializer=None,
                 n_iter: int = 1000,
                 n_iter_inner: int = 20,
                 use_line_search: bool = True,
                 lb: float = 1,
                 lg: float = 1,
                 nnz_tbeta: int = 3,
                 nnz_tgamma: int = 3,
                 logger_keys: Set = ('converged',)):
        """
        Initializes the model.

        Parameters
        ----------
        tol : float
            Tolerance for stopping criterion: ||tβ_{k+1} - tβ_k|| <= tol and ||t𝛄_{k+1} - t𝛄_k|| <= tol.

        tol_inner : float
            Tolerance for inner optimization subroutine (min ℒ w.r.t. 𝛄) stopping criterion:
            ||projected ∇ℒ|| <= tol_inner

        solver : {'pgd'}
            Solver to use in computational routines:
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
            Whether to use line search when optimizing w.r.t. 𝛄. If true, it starts from step_len = 1 and cuts it in half
            until the descent criterion is met. If false, it uses a fixed step size of 1/iteration_number.

        lb : float
            Regularization coefficient for the tβ-related term, see the loss-function description.

        lg : float
            Regularization coefficient for the t𝛄-related term, see the loss-function description.

        nnz_tbeta : int,
            How many non-zero coefficients are allowed in tβ.

        nnz_tgamma : int,
            How many non-zero coefficients are allowed in t𝛄.
        """

        self.tol = tol
        self.tol_inner = tol_inner
        self.solver = solver
        self.initializer = initializer
        self.n_iter = n_iter
        self.n_iter_inner = n_iter_inner
        self.use_line_search = use_line_search
        self.lb = lb
        self.lg = lg
        self.nnz_tbeta = nnz_tbeta
        self.nnz_tgamma = nnz_tgamma
        self.logger_keys = logger_keys

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            columns_labels: np.ndarray = None,
            initial_parameters: dict = None,
            warm_start=False,
            random_intercept=True,
            **kwargs):
        """
        Fits a Linear Model with Linear Mixed-Effects to the given data

        Parameters
        ----------
        x : np.ndarray
            Data. If columns_labels = None then it's assumed that columns_labels are in the first row of x.

        y : np.ndarray
            Answers, real-valued array.

        columns_labels : np.ndarray
            List of column labels: 1 -- fixed effect, 2 -- random effect, 3 -- both fixed and random,
                    0 -- groups labels, 4 -- answers standard deviations. There shall be only one
                    column of group labels and answers STDs, and overall n columns with fixed effects (1 or 3)
                    and k columns of random effects (2 or 3).

        initial_parameters: dict with possible fields:
            - 'beta0' : np.ndarray, shape = [n]
                Initial estimate of fixed effects. If None then it defaults to an all-ones vector.
            - 'gamma0' : np.ndarray, shape = [k]
                Initial estimate of random effects covariances. If None then it defaults to an all-ones vector.
            - 'tbeta0' : np.ndarray, shape = [n]
                Initial estimate of sparse fixed effects. If None then it defaults to an all-zeros vector.
            - 'tgamma0' : np.ndarray, shape = [k]
                Initial estimate of sparse random covariances. If None then it defaults to an all-zeros vector.

        warm_start : bool, default = False
            Whether to use previous parameters as initial ones. Overrides initial_parameters if given.
            Throws NotFittedError if set to True when not fitted.

        random_intercept : bool, default = True
            Whether treat the intercept as a random effect.
        kwargs :
            Not used currently, left here for passing debugging parameters.

        Returns
        -------
        self :
            Fitted regression model.
        """

        problem, _ = LinearLMEProblem.from_x_y(x, y, columns_labels, random_intercept=random_intercept, **kwargs)
        if initial_parameters is None:
            initial_parameters = {}
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
        num_fixed_effects = problem.num_features
        num_random_effects = problem.num_random_effects
        assert num_fixed_effects >= self.nnz_tbeta
        assert num_random_effects >= self.nnz_tgamma
        old_oracle = OldOracle(problem, lb=self.lb, lg=self.lg, k=self.nnz_tbeta, j=self.nnz_tgamma)

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
            beta = oracle.optimal_beta(gamma, tbeta)
            us = oracle.optimal_random_effects(beta, gamma)
            gamma = np.sum(us ** 2, axis=0) / oracle.problem.num_studies

        def projected_gradient(current_gamma, current_gradient_gamma):
            proj_gradient = current_gradient_gamma.copy()
            for j, _ in enumerate(current_gamma):
                if current_gamma[j] == 0 and current_gradient_gamma[j] <= 0:
                    proj_gradient[j] = 0
            return proj_gradient

        loss = oracle.loss(beta, gamma, tbeta, tgamma)
        gradient_gamma = oracle.gradient_gamma(beta, gamma, tgamma)
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
                while (np.linalg.norm(projected_gradient(gamma, gradient_gamma)) > self.tol_inner
                       and inner_iteration < self.n_iter_inner):
                    beta = oracle.optimal_beta(gamma, tbeta)
                    gradient_gamma = oracle.gradient_gamma(beta, gamma, tgamma)
                    # projecting the gradient to the set of constraints
                    direction = -projected_gradient(gamma, gradient_gamma)
                    if self.use_line_search:
                        # line search method
                        step_len = 0.1
                        for i, _ in enumerate(gamma):
                            if direction[i] < 0:
                                step_len = min(-gamma[i]/direction[i], step_len)

                        current_loss = oracle.loss(beta, gamma, tbeta, tgamma)
                        while oracle.loss(beta, gamma + step_len * direction, tbeta, tgamma) >= current_loss:
                            step_len *= 0.5
                            if step_len <= 1e-15:
                                break
                    else:
                        # fixed step size
                        step_len = 1 / iteration
                    gamma = gamma + step_len * direction
                    gradient_gamma = oracle.gradient_gamma(beta, gamma, tgamma)
                    inner_iteration += 1

                prev_tbeta = tbeta
                prev_tgamma = tgamma
                tbeta = oracle.optimal_tbeta(beta)
                tgamma = oracle.optimal_tgamma(tbeta, gamma)

            loss = oracle.loss(beta, gamma, tbeta, tgamma)
            if len(self.logger_keys) > 0:
                self.logger_.log(locals())

        us = oracle.optimal_random_effects(beta, gamma)

        per_cluster_coefficients = np.zeros((problem.num_studies, len(problem.column_labels)))

        for i, u in enumerate(us):
            fixed_effects_counter = 0
            random_effects_counter = 0

            for j, label in enumerate(problem.column_labels):
                if label == 1:
                    per_cluster_coefficients[i, j] = beta[fixed_effects_counter]
                    fixed_effects_counter += 1
                elif label == 2:
                    per_cluster_coefficients[i, j] = u[random_effects_counter]
                    random_effects_counter += 1
                elif label == 3:
                    per_cluster_coefficients[i, j] = beta[fixed_effects_counter] + u[random_effects_counter]
                    fixed_effects_counter += 1
                    random_effects_counter += 1
                else:
                    continue

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
