import numpy as np
from skmixed.linear.problems import LinearProblem


class LinearOracle:

    def __init__(self, problem: LinearProblem):
        assert problem.c is None, "non-trivial C is not supported by a LinearOracle. Use LinearOralceSR3 instead."
        self.problem = problem

    def loss(self, x):
        return 1 / 2 * np.linalg.norm(self.problem.a.dot(x) - self.problem.b, ord=2) ** 2

    def gradient(self, x):
        return self.problem.a.T.dot(self.problem.a.dot(x) - self.problem.b)

    def hessian(self, x):
        return self.problem.a.T.dot(self.problem.a)

    def value_function(self, x):
        return self.loss(x)

    def gradient_value_function(self, x):
        return self.gradient(x)


class LinearOracleSR3:

    def __init__(self, problem: LinearProblem, lam=1, practical=False):
        a = problem.a
        c = problem.c
        self.problem = problem
        self.lam = lam
        self.h_matrix = a.T.dot(a) + lam * c.dot(c)
        self.h_inv = np.linalg.inv(self.h_matrix)
        self.ab = a.T.dot(problem.b)
        self.practical = practical
        if practical:
            self.f_matrix = np.vstack([lam * a.dot(self.h_inv).dot(c.T),
                                       (np.sqrt(lam) * (np.eye(c.shape[0]) - lam * c.dot(self.h_inv).dot(c.T)))])
            self.g_matrix = np.vstack([np.eye(a.shape[0]) - a.dot(self.h_inv).dot(a.T),
                                       np.sqrt(lam) * c.dot(self.h_inv).dot(a)])
            self.g = self.g_matrix.dot(problem.b)

    def loss(self, x, w):
        return (1 / 2 * np.linalg.norm(self.problem.a.dot(x) - self.problem.b, ord=2) ** 2 +
                self.lam / 2 * np.linalg.norm(self.problem.c.dot(x) - w, ord=2) ** 2)

    def value_function(self, x):
        assert not self.practical, "The oracle is in 'practical' mode. The value function is inaccessible."
        return 1 / 2 * np.linalg.norm(self.f_matrix.dot(x) - self.g, ord=2) ** 2

    def gradient_value_function(self, x):
        assert not self.practical, "The oracle is in 'practical' mode. The value function is inaccessible."
        return self.f_matrix.T.dot(self.f_matrix.dot(x) - self.g, ord=2)

    def find_optimal_parameters(self, x0, regularizer=None, tol=1e-4, max_iter=1000, **kwargs):
        x = x0
        step_len = 1 / self.lam
        x_prev = np.infty
        iteration = 0

        while np.linalg.norm(x - x_prev) > tol and iteration < max_iter:
            x_prev = x
            y = self.h_inv(self.ab + self.lam * self.problem.c.T.dot(x))
            x = regularizer.prox(y, step_len)
            iteration += 1

        return x
