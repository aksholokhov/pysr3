import numpy as np

from pysr3.linear.problems import LinearProblem
from pysr3.priors import Prior, NonInformativePrior


class LinearOracle:
    """
    Implements a supplementary class that abstracts model. That is, it takes a problem
    and provides losses and gradients with respect to the parameters of the model.

    It separates the model form the optimization routine for better code patterns.
    The solver takes an oracle and optimizes its loss using its gradient, but it does not know which model it optimizes.
    The oracle, in its turn, has no idea how the solution for its model will be obtained.
    """

    def __init__(self, problem: LinearProblem = None, prior: Prior = None):
        """
        Initializes LinearOracle

        Parameters
        ----------
        problem: LinearProblem, optional
            an instance of LinearProblem containing the data
        prior: Prior
            an instance of Prior for the models' coefficients, if needed. See the docs for pysr3.priors module.
        """
        self.problem = problem
        self.prior = prior if prior else NonInformativePrior()

    def instantiate(self, problem):
        """
        Attaches the given problem to the oracle

        Parameters
        ----------
        problem: LinearProblem
            instance of the problem

        Returns
        -------
        oracle: LinearOracle
            oracle for this problem
        """
        self.problem = problem
        self.prior.instantiate(problem)

    def forget(self):
        """
        Detaches the problem from the oracle
        """
        self.problem = None
        self.prior.forget()

    def loss(self, x):
        """
        Calculates Gaussian negative log-likelihood for the given set of models parameters x.

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        loss: float
            loss value
        """
        return 1 / 2 * np.linalg.norm(self.problem.a.dot(x) - self.problem.b, ord=2) ** 2 + self.prior.loss(x)

    def gradient(self, x):
        """
        Calculates gradient of Gaussian negative log-likelihood with respect to the given set of models parameters x.

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        gradient: ndarray (num_features, )
            gradient
        """
        return self.problem.a.T.dot(self.problem.a.dot(x) - self.problem.b) + self.prior.gradient(x)

    def hessian(self, x):
        """
        Calculates Hessian of Gaussian negative log-likelihood with respect to the given set of models parameters x.

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        hessian: ndarray (num_features, num_features)
            Hessian
        """
        return self.problem.a.T.dot(self.problem.a) + self.prior.hessian(x)

    def value_function(self, x):
        """
        Calculates value function for the given set of models parameters x. It's the same as loss
        if the oracle does not implement an SR3 relaxation.

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        loss: float
            loss value
        """
        return self.loss(x)

    def gradient_value_function(self, x):
        """
        Calculates gradient of the value function with respect to the given set of models parameters x.
        It's the same as normal gradient if the oracle does not implement an SR3 relaxation.

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        gradient: ndarray (num_features, )
            gradient
        """
        return self.gradient(x)

    def aic(self, x):
        """
        Calculates Akaike information criterion (AIC)

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        aic: float
            AIC
        """
        p = sum(x != 0)
        return self.loss(x) + 2 * p

    def bic(self, x):
        """
        Calculates Bayess information criterion (BIC)

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        bic: float
            BIC
        """
        p = sum(x != 0)
        return self.loss(x) + np.log(self.problem.num_objects) * p


class LinearOracleSR3:
    """
    Implements a supplementary class that abstracts SR3-model. That is, it takes a problem
    and provides losses and gradients with respect to the parameters of the model.

    It separates the model form the optimization routine for better code patterns.
    The solver takes an oracle and optimizes its loss using its gradient, but it does not know which model it optimizes.
    The oracle, in its turn, has no idea how the solution for its model will be obtained.
    """

    def __init__(self, problem: LinearProblem = None, lam: float = 1, practical: bool = False, prior: Prior = None):
        """
        Instantiates an oracle

        Parameters
        ----------
        problem: LinearProblem, optional
            an instance of LinearProblem containing the data
        prior: Prior
            an instance of Prior for the models' coefficients, if needed. See the docs for pysr3.priors module.
        lam: float
            coefficient for the strength SR3-relaxation. It's NOT the same as the regularization (sparsity)
            coefficient. See the paper for more details.
        practical: bool
            whether to use an optimization method that is much faster than the default.
        """
        assert not prior, "Priors for LinearOracleSR3 are not supported yet"
        self.prior = prior if prior else NonInformativePrior()
        self.lam = lam
        self.practical = practical
        self.problem = problem
        self.f_matrix = None
        self.g_matrix = None
        self.h_matrix = None
        self.h_inv = None
        self.g = None
        self.ab = None

    def instantiate(self, problem):
        """
        Attaches the given problem to the oracle

        Parameters
        ----------
        problem: LinearProblem
            instance of the problem

        Returns
        -------
        oracle: LinearOracleSR3
            oracle for this problem
        """
        self.problem = problem
        a = problem.a
        c = problem.c
        lam = self.lam
        self.h_matrix = a.T.dot(a) + lam * c.dot(c)
        self.h_inv = np.linalg.inv(self.h_matrix)
        self.ab = a.T.dot(problem.b)
        if not self.practical:
            self.f_matrix = np.vstack([lam * a.dot(self.h_inv).dot(c.T),
                                       (np.sqrt(lam) * (np.eye(c.shape[0]) - lam * c.dot(self.h_inv).dot(c.T)))])
            self.g_matrix = np.vstack([np.eye(a.shape[0]) - a.dot(self.h_inv).dot(a.T),
                                       np.sqrt(lam) * c.dot(self.h_inv).dot(a.T)])
            self.g = self.g_matrix.dot(problem.b)

    def forget(self):
        """
        Detaches the problem from the oracle
        """
        self.problem = None
        self.f_matrix = None
        self.g_matrix = None
        self.h_matrix = None
        self.h_inv = None
        self.g = None
        self.ab = None

    def loss(self, x, w):
        """
        Calculates Gaussian negative log-likelihood of SR3 relaxation for the given set of models parameters x.

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters
        w: ndarray (num_features, )
            dual (relaxed) parameters that SR3-relaxation introduces
        Returns
        -------
        loss: float
            loss value
        """

        return (1 / 2 * np.linalg.norm(self.problem.a.dot(x) - self.problem.b, ord=2) ** 2 +
                self.lam / 2 * np.linalg.norm(self.problem.c.dot(x) - w, ord=2) ** 2) + self.prior.loss(x)

    def value_function(self, x):
        """
        Calculates value function for the given set of models parameters x.

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        loss: float
            loss value
        """
        assert not self.practical, "The oracle is in 'practical' mode. The value function is inaccessible."
        return 1 / 2 * np.linalg.norm(self.f_matrix.dot(x) - self.g, ord=2) ** 2

    def gradient_value_function(self, x):
        """
        Calculates gradient of the value function with respect to the given set of models parameters x.
        It's the same as normal gradient if the oracle does not implement an SR3 relaxation.

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        gradient: ndarray (num_features, )
            gradient
        """
        assert not self.practical, "The oracle is in 'practical' mode. The value function is inaccessible."
        return self.f_matrix.T.dot(self.f_matrix.dot(x) - self.g)

    def find_optimal_parameters(self, x0, regularizer=None, tol: float = 1e-4, max_iter: int = 1000, **kwargs):
        """
        Implements a "practical" optimization scheme that works faster than just a standard gradient descent.
        This function is meant to be called by pysr3.solvers.FakePGDSolver

        Parameters
        ----------
        x0: ndarray (num_features, )
            starting point for the optimization
        regularizer: Regularizer
            regularizer that implements sparsifiction prior.
        tol: float
            tolerance for the solver
        max_iter: int
            maximum number of iterations
        kwargs:
            other keyword arguments

        Returns
        -------
        x: ndarray (num_features, )
            the optimal solution
        """
        x = x0
        step_len = 1 / self.lam
        x_prev = np.infty
        iteration = 0

        while np.linalg.norm(x - x_prev) > tol and iteration < max_iter:
            x_prev = x
            y = self.h_inv.dot(self.ab + self.lam * self.problem.c.T.dot(x))
            x = regularizer.prox(y, step_len)
            iteration += 1

        return x

    def aic(self, x):
        """
        Calculates Akaike information criterion (AIC)

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        aic: float
            AIC
        """
        p = sum(x != 0)
        oracle = LinearOracle(self.problem, self.prior)
        return oracle.loss(x) + 2 * p

    def bic(self, x):
        """
        Calculates Bayess information criterion (BIC)

        Parameters
        ----------
        x: ndarray (num_features, )
            models parameters

        Returns
        -------
        bic: float
            BIC
        """
        p = sum(x != 0)
        oracle = LinearOracle(self.problem, self.prior)
        return oracle.loss(x) + np.log(self.problem.num_objects) * p
