import unittest

from pysr3.priors import GaussianPrior


class TestPriors(unittest.TestCase):

    def test_gaussian_prior(self):
        prior = GaussianPrior(params={"intercept": (0, 2)})
        prior.instantiate(problem_columns=["intercept"])
        self.assertEqual(prior.loss(2), 1)
        self.assertEqual(prior.gradient(2)[0], 1)
        self.assertEqual(prior.hessian(2)[0], 1 / 2)
        prior.forget()
        self.assertIsNone(prior.weights)
