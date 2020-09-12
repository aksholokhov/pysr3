import unittest
from unittest import TestCase

import numpy as np
from numpy import allclose
from scipy.misc import derivative

from skmixed.lme.oracles import LinearLMEOracle, LinearLMEOracleRegularized
from skmixed.legacy.oracles import LinearLMEOracle as OldOracle
from skmixed.lme.problems import LinearLMEProblem
from skmixed.helpers import random_effects_to_matrix


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

    def test_no_data_problem(self):
        random_seed = 43
        problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[10, 10, 10],
                                                             features_labels=[],
                                                             random_intercept=True,
                                                             seed=random_seed)
        beta = true_parameters['beta']
        us = random_effects_to_matrix(true_parameters['random_effects'])
        empirical_gamma = np.sum(us ** 2, axis=0) / problem.num_groups
        rtol = 1e-1
        atol = 1e-1
        oracle = LinearLMEOracle(problem)

        maybe_beta = oracle.optimal_beta(empirical_gamma)
        maybe_us = oracle.optimal_random_effects(maybe_beta, empirical_gamma)
        self.assertTrue(allclose(maybe_beta + maybe_us, beta + us, rtol=rtol, atol=atol),
                        msg="No-data-problem is not right")
        return None

    def test_non_regularized_oracle_is_zero_regularized_oracle(self):
        num_fixed_effects = 4
        num_random_effects = 3
        problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[4, 5, 10],
                                                             features_labels=[3, 3, 1, 2],
                                                             random_intercept=False,
                                                             obs_std=0.1,
                                                             seed=42)
        # when both regularization coefficients are zero, these two oracles should be exactly equivalent
        oracle_non_regularized = LinearLMEOracle(problem)
        oracle_regularized = LinearLMEOracleRegularized(problem, lg=0, lb=0, nnz_tbeta=1, nnz_tgamma=1)
        np.random.seed(42)
        trials = 100
        rtol = 1e-14
        atol = 1e-14
        for random_beta, random_gamma, random_tbeta, random_tgamma in zip(np.random.rand(trials, num_fixed_effects),
                                                                          np.random.rand(trials, num_random_effects),
                                                                          np.random.rand(trials, num_fixed_effects),
                                                                          np.random.rand(trials, num_random_effects),
                                                                          ):
            loss1 = oracle_regularized.loss(random_beta, random_gamma, random_tbeta, random_tgamma)
            loss2 = oracle_non_regularized.loss(random_beta, random_gamma)
            self.assertAlmostEqual(loss1, loss2, delta=atol,
                                   msg="Loss of zero-regularized and non-regularized oracles is different")
            gradient1 = oracle_regularized.gradient_gamma(random_beta, random_gamma, random_tgamma)
            gradient2 = oracle_non_regularized.gradient_gamma(random_beta, random_gamma)
            self.assertTrue(allclose(gradient1, gradient2, rtol=rtol, atol=atol),
                            msg="Gradients w.r.t. gamma of zero-regularized and non-regularized oracles are different")
            hessian1 = oracle_regularized.hessian_gamma(random_beta, random_gamma)
            hessian2 = oracle_non_regularized.hessian_gamma(random_beta, random_gamma)
            self.assertTrue(allclose(hessian1, hessian2, rtol=100 * rtol, atol=100 * atol),
                            msg="Hessian w.r.t. gamma of zero-regularized and non-regularized oracles are different")
            beta1 = oracle_regularized.optimal_beta(random_gamma, random_tbeta)
            beta2 = oracle_non_regularized.optimal_beta(random_gamma)
            self.assertTrue(allclose(beta1, beta2, rtol=rtol, atol=atol),
                            msg="Optimal betas of zero-regularized and non-regularized oracles are different")
            us1 = oracle_regularized.optimal_random_effects(random_beta, random_gamma)
            us2 = oracle_non_regularized.optimal_random_effects(random_beta, random_gamma)
            self.assertTrue(allclose(us1, us2, rtol=rtol, atol=atol),
                            msg="Optimal random effects of zero-regularized and non-regularized oracles is different")
        return None

    def test_beta_to_gamma_map(self):
        problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[4, 5, 10],
                                                             features_labels=[3, 3, 1, 2, 3, 1, 2],
                                                             random_intercept=False,
                                                             obs_std=0.1,
                                                             seed=42)
        oracle = LinearLMEOracle(problem)
        true_beta_to_gamma_map = np.array([-1, 0, 1, -1, 3, -1])
        for e1, e2 in zip(true_beta_to_gamma_map, oracle.beta_to_gamma_map):
            self.assertEqual(e1, e2, msg="Beta-to-gamma mask is not right: \n %s is not \n %s as should be" % (
                true_beta_to_gamma_map,
                oracle.beta_to_gamma_map
            ))

    def test_jones2010n_eff(self):
        # This test is based on the fact that
        # in case of a random intercept model the n_eff can be represented through intraclass correlation rho.
        # See original Jones2010 paper for more details.
        for seed in range(10):
            problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[40, 30, 50],
                                                                 features_labels=[],
                                                                 random_intercept=True,
                                                                 obs_std=0.1,
                                                                 seed=seed)
            oracle = LinearLMEOracle(problem)
            gamma = true_parameters['gamma']
            rho = gamma/(gamma + 0.1)
            oracle._recalculate_cholesky(true_parameters['gamma'])
            n_eff = oracle._jones2010n_eff()
            assert np.allclose(n_eff, sum([ni/(1+(ni-1)*rho) for ni in problem.groups_sizes]))

    def test_hodges2001ddf(self):
        # From here:
        # https://www.jstor.org/stable/2673485?seq=1
        problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[40, 30, 50],
                                                             features_labels=[3, 3, 3],
                                                             random_intercept=True,
                                                             obs_std=0.1,
                                                             seed=42)
        oracle = LinearLMEOracle(problem)
        true_gamma = true_parameters['gamma']
        ddf = oracle._hodges2001ddf(true_gamma)
        #  #|beta| <= DDoF <= #|beta| + num_groups*#|u|
        assert 4 <= ddf <= 4+4*3

    def test_hat_matrix(self):
        for seed in range(10):
            problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[40, 30, 50],
                                                                 features_labels=[3, 3],
                                                                 random_intercept=True,
                                                                 obs_std=0.1,
                                                                 seed=seed)
            oracle = LinearLMEOracle(problem)
            gamma = true_parameters['gamma']
            optimal_beta = oracle.optimal_beta(gamma)
            us = oracle.optimal_random_effects(optimal_beta, gamma)
            ys_true = []
            ys_optimal_true = []
            for (x, y, z, l), u in zip(problem, us):
                ys_optimal_true.append(x.dot(optimal_beta) + z.dot(u))
                ys_true.append(y)
            ys_true = np.concatenate(ys_true)
            ys_optimal_true = np.concatenate(ys_optimal_true)
            hat_matrix = oracle._hat_matrix(gamma)
            ys_optimal_hat = hat_matrix.dot(ys_true)
            assert np.allclose(ys_optimal_true, ys_optimal_hat)




if __name__ == '__main__':
    unittest.main()
