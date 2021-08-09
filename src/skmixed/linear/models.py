from typing import Set

import numpy as np
from skmixed.linear.problems import LinearProblem
from skmixed.logger import Logger


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
            random_intercept=True,
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

                        -   | 'beta0' : np.ndarray, shape = [n],
                            | Initial estimate of fixed effects. If None then it defaults to an all-ones vector.

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
            an instance of LinearLMEProblem that contains all data-dependent information

        initial_parameters : np.ndarray
            Dict with possible fields:

                -   | 'beta0' : np.ndarray, shape = [n],
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

        x = initial_parameters.get("x", np.ones(problem.num_features))

        self.logger_ = Logger(self.logger_keys)

        optimal_x = self.solver.optimize(x, oracle=self.oracle, regularizer=self.regularizer, logger=self.logger_)

        self.coef_ = {
            "x": optimal_x,
        }

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
