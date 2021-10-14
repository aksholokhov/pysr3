import unittest
from unittest import TestCase

import numpy as np
from numpy import allclose

from pysr3.glms.problems import PoissonProblem
from pysr3.glms.oracles import GLMOracle
from pysr3.glms.link_functions import PoissonLinkFunction


class TestGLMOracle(TestCase):

    def test_gradient(self):
        trials = 100
        random_seed = 34
        r = 1e-6
        rtol = 1e-4
        atol = 1e-5
        problem = PoissonProblem.generate(seed=random_seed)
        oracle = GLMOracle(problem=problem, prior=None, link_function=PoissonLinkFunction())
        np.random.seed(random_seed)
        for j in range(trials):
            with self.subTest(j=j):
                x = np.random.rand(problem.num_features)
                dx = np.random.rand(problem.num_features)
                gradient = oracle.gradient(x)
                maybe_dir = gradient.dot(dx)
                true_dir = (oracle.loss(x + r * dx)
                            - oracle.loss(x - r * dx)
                            ) / (2 * r)
                self.assertTrue(allclose(maybe_dir, true_dir, rtol=rtol, atol=atol),
                                msg="Gradient x does not look right")

    def test_hessian(self):
        trials = 100
        random_seed = 34
        r = 1e-6
        rtol = 1e-4
        atol = 1e-5
        problem = PoissonProblem.generate(seed=random_seed)
        oracle = GLMOracle(problem=problem, prior=None, link_function=PoissonLinkFunction())
        np.random.seed(random_seed)

        for j in range(trials):
            with self.subTest(j=j):
                np.random.seed(random_seed + j)
                x = np.random.rand(problem.num_features)
                dx = np.random.rand(problem.num_features)
                hess = oracle.hessian(x)
                maybe_dir = hess.dot(dx)
                true_dir = (oracle.gradient(x + r * dx)
                            - oracle.gradient(x - r * dx)
                            ) / (2 * r)
                self.assertTrue(allclose(maybe_dir, true_dir, rtol=rtol, atol=atol),
                                msg="Hessian does not look right")