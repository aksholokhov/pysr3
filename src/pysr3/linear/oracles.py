import numpy as np

from pysr3.linear.problems import LinearProblem
from pysr3.priors import Prior, NonInformativePrior


class LinearOracle:

    def __init__(self, problem: LinearProblem = None, prior: Prior = None):
        self.problem = problem
        self.prior = prior if prior else NonInformativePrior()

    def instantiate(self, problem):
        self.problem = problem
        self.prior.instantiate(problem)

    def forget(self):
        self.problem = None
        self.prior.forget()

    def loss(self, x):
        return 1 / 2 * np.linalg.norm(self.problem.a.dot(x) - self.problem.b, ord=2) ** 2 + self.prior.loss(x)

    def gradient(self, x):
        return self.problem.a.T.dot(self.problem.a.dot(x) - self.problem.b) + self.prior.gradient(x)

    def hessian(self, x):
        return self.problem.a.T.dot(self.problem.a) + self.prior.hessian(x)

    def value_function(self, x):
        return self.loss(x)

    def gradient_value_function(self, x):
        return self.gradient(x)

    def aic(self, x):
        p = sum(x != 0)
        return self.loss(x) + 2 * p

    def bic(self, x):
        p = sum(x != 0)
        return self.loss(x) + np.log(self.problem.num_objects) * p


class LinearOracleSR3:

    def __init__(self, problem: LinearProblem = None, lam=1, practical=False, prior: Prior = None):
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
        self.problem = None
        self.f_matrix = None
        self.g_matrix = None
        self.h_matrix = None
        self.h_inv = None
        self.g = None
        self.ab = None

    def loss(self, x, w):
        return (1 / 2 * np.linalg.norm(self.problem.a.dot(x) - self.problem.b, ord=2) ** 2 +
                self.lam / 2 * np.linalg.norm(self.problem.c.dot(x) - w, ord=2) ** 2) + self.prior.loss(x)

    def value_function(self, x):
        assert not self.practical, "The oracle is in 'practical' mode. The value function is inaccessible."
        return 1 / 2 * np.linalg.norm(self.f_matrix.dot(x) - self.g, ord=2) ** 2

    def gradient_value_function(self, x):
        assert not self.practical, "The oracle is in 'practical' mode. The value function is inaccessible."
        return self.f_matrix.T.dot(self.f_matrix.dot(x) - self.g)

    def find_optimal_parameters(self, x0, regularizer=None, tol=1e-4, max_iter=1000, **kwargs):
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
        p = sum(x != 0)
        oracle = LinearOracle(self.problem, self.prior)
        return oracle.loss(x) + 2 * p

    def bic(self, x):
        p = sum(x != 0)
        oracle = LinearOracle(self.problem, self.prior)
        return oracle.loss(x) + np.log(self.problem.num_objects) * p
