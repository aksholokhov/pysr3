import unittest
from unittest import TestCase

import numpy as np
from numpy import allclose
from scipy.misc import derivative

from lib.new_oracles import LinearLMEOracle
from lib.legacy.oracles import LinearLMEOracle as OldOracle
from lib.problems import LinearLMEProblem


class TestLinearLMEOracle(TestCase):

    def test_compare_to_old_oracle(self):
        num_fixed_effects = 4
        num_random_effects = 2
        problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[4, 5, 10],
                                                             features_labels=[3, 3, 1],
                                                             random_intercept=False,
                                                             obs_std=0.1,
                                                             seed=42)
        new_oracle = LinearLMEOracle(problem)
        old_oracle = OldOracle(problem)
        np.random.seed(42)
        trials = 100
        # the error should stem only from Cholesky/regular inversions instabilities, so
        # tolerances should pretty much represent machine precision
        rtol = 1e-8
        atol = 1e-10
        for random_beta, random_gamma in zip(np.random.rand(trials, num_fixed_effects),
                                             np.random.rand(trials, num_random_effects)):
            loss1 = new_oracle.loss(random_beta, random_gamma)
            loss2 = old_oracle.loss(random_beta, random_gamma)
            self.assertAlmostEqual(loss1, loss2, delta=atol, msg="Loss does not match with old oracle")
            gradient1 = new_oracle.gradient_gamma(random_beta, random_gamma)
            gradient2 = old_oracle.gradient_gamma(random_beta, random_gamma)
            self.assertTrue(allclose(gradient1, gradient2, rtol=rtol, atol=atol),
                            msg="Gradients don't match with old oracle")
            hessian1 = new_oracle.hessian_gamma(random_beta, random_gamma)
            hessian2 = old_oracle.hessian_gamma(random_beta, random_gamma)
            self.assertTrue(allclose(hessian1, hessian2, rtol=100 * rtol, atol=100 * atol),
                            msg="Hessian does not match with old oracle")
            beta1 = new_oracle.optimal_beta(random_gamma)
            beta2 = old_oracle.optimal_beta(random_gamma)
            self.assertTrue(allclose(beta1, beta2, rtol=rtol, atol=atol),
                            msg="Optimal betas don't match with old oracle")
            us1 = new_oracle.optimal_random_effects(random_beta, random_gamma)
            us2 = old_oracle.optimal_random_effects(random_beta, random_gamma)
            self.assertTrue(allclose(us1, us2, rtol=rtol, atol=atol),
                            msg="Optimal random effects don't match with old oracle")
        return None

    def test_gamma_derivatives(self):
        trials = 5
        rtol = 1e-3
        atol = 1e-2
        dx = rtol / 1000
        for random_seed in np.random.randint(0, 1000, size=trials):
            np.random.seed(random_seed)
            problem, true_parameters = LinearLMEProblem.generate(features_labels=[3, 3],
                                                                 random_intercept=False,
                                                                 seed=random_seed)
            beta = true_parameters['beta']
            oracle = LinearLMEOracle(problem)
            points = np.random.rand(30, 2)
            beta = np.random.rand(len(beta))

            oracle_gradient = np.array([oracle.gradient_gamma(beta, g) for g in points])
            partial_derivative_1 = np.array(
                [derivative(lambda x: oracle.loss(beta, np.array([x, g[1]])), g[0], dx=dx) for g in points])
            partial_derivative_2 = np.array(
                [derivative(lambda x: oracle.loss(beta, np.array([g[0], x])), g[1], dx=dx) for g in points])
            for i, (a, c, d, e) in enumerate(zip(points, oracle_gradient, partial_derivative_1, partial_derivative_2)):
                self.assertTrue(allclose(c[0], d, rtol=rtol, atol=atol),
                                msg="Gamma gradient does not match with numerical partial derivative: %d" % i)
                self.assertTrue(allclose(c[1], e, rtol=rtol, atol=atol),
                                msg="Gamma gradient does not match with numerical partial derivative: %d" % i)
        return None

    def test_hessian_gamma(self):
        trials = 100
        random_seed = 34
        r = 1e-6
        rtol = 1e-5
        atol = 1e-7
        problem, true_parameters = LinearLMEProblem.generate(seed=random_seed)
        oracle = LinearLMEOracle(problem)
        np.random.seed(random_seed)
        for j in range(trials):
            beta = np.random.rand(problem.num_fixed_effects)
            gamma = np.random.rand(problem.num_random_effects)
            dg = np.random.rand(problem.num_random_effects)
            hess = oracle.hessian_gamma(beta, gamma)
            maybe_dir = hess.dot(dg)
            true_dir = (oracle.gradient_gamma(beta, gamma + r * dg)
                        - oracle.gradient_gamma(beta, gamma - r * dg)
                        ) / (2 * r)

            self.assertTrue(allclose(maybe_dir, true_dir, rtol=rtol, atol=atol), msg="Hessian does not look right")

    # # We exclude this test because it does not make sense to test beta ang random_effects separately (when X = Z they
    # # are identifiable only in the limit
    # def test_optimal_beta(self):
    #     trials = 100
    #     num_features = 5
    #     num_random_effects = 3
    #     for random_seed in np.random.randint(0, 1000, size=trials):
    #         np.random.seed(random_seed)
    #         noise_variance = 1e-2
    #         rtol = 1e-5
    #         atol = noise_variance
    #         problem, beta, gamma, us, err = LinearLMEProblem.generate(study_sizes=[50, 50, 50, 50],
    #                                                                   num_features=num_features,
    #                                                                   num_random_effects=num_random_effects,
    #                                                                   seed=random_seed)
    #         oracle = LinearLMEOracle(problem)
    #         maybe_beta = oracle.optimal_beta(gamma)
    #         maybe_us = oracle.optimal_random_effects(maybe_beta, gamma)
    #         self.assertAlmostEqual(allclose(maybe_beta + maybe_us, beta + us, rtol=rtol, atol=atol),
    #                                msg="Optimal beta is not right for a random problem with seed=%d" % random_seed)

    def test_no_data_problem(self):
        random_seed = 43
        problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[10, 10, 10],
                                                             features_labels=[],
                                                             random_intercept=True,
                                                             seed=random_seed)
        beta = true_parameters['beta']
        gamma = true_parameters['gamma']
        us = true_parameters['random_effects']
        rtol = 1e-1
        atol = 1e-1
        oracle = LinearLMEOracle(problem)

        maybe_beta = oracle.optimal_beta(gamma)
        maybe_us = oracle.optimal_random_effects(maybe_beta, gamma)
        self.assertTrue(allclose(maybe_beta + maybe_us, beta + us, rtol=rtol, atol=atol),
                        msg="No-data-problem is not right")
        return None

    # def test_optimal_random_effects(self):
    #     trials = 100
    #     noise_variance = 1e-2
    #     for random_seed in np.random.randint(0, 1000, size=trials):
    #         np.random.seed(random_seed)
    #
    #         problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(study_sizes=[20, 30, 50],
    #                                                                                    obs_std=noise_variance,
    #                                                                                    seed=random_seed)
    #         oracle = LinearLMEOracle(problem)
    #         maybe_random_effects = oracle.optimal_random_effects(beta, gamma)
    #         self.assertTrue(allclose(random_effects, maybe_random_effects))


if __name__ == '__main__':
    unittest.main()
