import numpy as np

from pysr3.linear.problems import LinearProblem
from pysr3.priors import Prior, NonInformativePrior
from pysr3.glms.link_functions import LinkFunction, IdentityLinkFunction


class GLMOracle:

    def __init__(self, problem: LinearProblem = None, prior: Prior = None, link_function: LinkFunction = None):
        self.problem = problem
        self.prior = prior if prior else NonInformativePrior()
        self.link_function = link_function if link_function else IdentityLinkFunction

    def instantiate(self, problem):
        self.problem = problem
        self.prior.instantiate(problem)

    def forget(self):
        self.problem = None
        self.prior.forget()

    def loss(self, x):
        a = self.problem.a
        b = self.problem.b
        return (1 / self.problem.obs_std ** 2 * (
                    self.link_function.value(a.dot(x)) - b * a.dot(x))).sum() + self.prior.loss(x)

    def gradient(self, x):
        a = self.problem.a
        b = self.problem.b
        return (a.T * (1 / self.problem.obs_std ** 2 * (
                    self.link_function.gradient(a.dot(x)) - b))).T + self.prior.gradient(x)

    def hessian(self, x):
        res = 0
        for i, ai in enumerate(self.problem.a):
            ai = ai.reshape(-1, 1)
            res = res + (1 / self.problem.obs_std[i] ** 2) * self.link_function.hessian(ai.dot(x)) * ai.dot(ai.T)
        return res + self.prior.hessian(x)

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

#
# class GLMOracleSR3:
#
#     def __init__(self, problem: LinearProblem = None, lam=1, practical=False, prior: Prior = None,
#                  link_function: Callable = None):
#         assert not prior, "Priors for LinearOracleSR3 are not supported yet"
#         self.prior = prior if prior else NonInformativePrior()
#         self.lam = lam
#         self.practical = practical
#         self.problem = problem
#
#     def instantiate(self, problem):
#         self.problem = problem
#         self.prior.instantiate(problem)
#
#     def forget(self):
#         self.problem = None
#         self.prior.forget()
#
#     def loss(self, x, w):
#         a = self.problem.a
#         b = self.problem.b
#         return ((1 / self.problem.obs_std ** 2 * (self.link_function.value(a.dot(x)) - b * a.dot(x))).sum()
#                 + self.lam / 2 * np.linalg.norm(x - w, ord=2) ** 2
#                 + self.prior.loss(x))
#
#     def value_function(self, w):
#         return 1 / 2 * np.linalg.norm(self.f_matrix.dot(x) - self.g, ord=2) ** 2
#
#     def gradient_value_function(self, x):
#         return self.f_matrix.T.dot(self.f_matrix.dot(x) - self.g)
#
#     def find_optimal_parameters(self, x0, regularizer=None, tol=1e-4, max_iter=1000, **kwargs):
#         x = x0
#         step_len = 1 / self.lam
#         x_prev = np.infty
#         iteration = 0
#
#         while np.linalg.norm(x - x_prev) > tol and iteration < max_iter:
#             x_prev = x
#             y = self.h_inv.dot(self.ab + self.lam * self.problem.c.T.dot(x))
#             x = regularizer.prox(y, step_len)
#             iteration += 1
#
#         return x
#
#     def aic(self, x):
#         p = sum(x != 0)
#         oracle = LinearOracle(self.problem, self.prior)
#         return oracle.loss(x) + 2 * p
#
#     def bic(self, x):
#         p = sum(x != 0)
#         oracle = LinearOracle(self.problem, self.prior)
#         return oracle.loss(x) + np.log(self.problem.num_objects) * p
