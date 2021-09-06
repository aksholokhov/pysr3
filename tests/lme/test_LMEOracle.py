import unittest
from unittest import TestCase

import numpy as np
from numpy import allclose

from pysr3.lme.oracles import LinearLMEOracle, LinearLMEOracleSR3
from pysr3.lme.problems import LMEProblem, FIXED, RANDOM, FIXED_RANDOM
from pysr3.lme.problems import random_effects_to_matrix


class TestLinearLMEOracle(TestCase):

    def test_gradients(self):
        trials = 100
        random_seed = 34
        r = 1e-6
        rtol = 1e-4
        atol = 1e-5
        problem, true_parameters = LMEProblem.generate(seed=random_seed)
        oracle = LinearLMEOracle(problem)
        np.random.seed(random_seed)
        for j in range(trials):
            with self.subTest(j=j):
                beta = np.random.rand(problem.num_fixed_features)
                gamma = np.random.rand(problem.num_random_features)
                db = np.random.rand(problem.num_fixed_features)
                gradient_beta = oracle.gradient_beta(beta, gamma)
                maybe_dir = gradient_beta.dot(db)
                true_dir = (oracle.loss(beta + r * db, gamma)
                            - oracle.loss(beta - r * db, gamma)
                            ) / (2 * r)
                self.assertTrue(allclose(maybe_dir, true_dir, rtol=rtol, atol=atol),
                                msg="Gradient beta does not look right")
                dg = np.random.rand(problem.num_random_features)
                gradient_gamma = oracle.gradient_gamma(beta, gamma)
                maybe_dir = gradient_gamma.dot(dg)
                true_dir = (oracle.loss(beta, gamma + r * dg)
                            - oracle.loss(beta, gamma - r * dg)
                            ) / (2 * r)
                self.assertTrue(allclose(maybe_dir, true_dir, rtol=rtol, atol=atol),
                                msg="Gradient gamma does not look right")

    def test_hessians(self):
        trials = 40
        random_seed = 34
        r = 1e-5
        rtol = 1e-4
        atol = 1e-4
        problem, true_parameters = LMEProblem.generate(groups_sizes=[5, 8, 10],
                                                       features_labels=[FIXED_RANDOM, FIXED, RANDOM],
                                                       seed=random_seed,
                                                       fit_fixed_intercept=True,
                                                       fit_random_intercept=True)
        oracle = LinearLMEOracle(problem)

        for j in range(trials):
            with self.subTest(j=j):
                np.random.seed(random_seed + j)

                beta = np.random.rand(problem.num_fixed_features)
                gamma = np.random.rand(problem.num_random_features)

                db = np.random.rand(problem.num_fixed_features)
                hess = oracle.hessian_beta(beta, gamma)
                maybe_dir = hess.dot(db)
                true_dir = (oracle.gradient_beta(beta + r * db, gamma)
                            - oracle.gradient_beta(beta - r * db, gamma)
                            ) / (2 * r)
                self.assertTrue(allclose(maybe_dir, true_dir, rtol=rtol, atol=atol),
                                msg="Hessian beta does not look right")

                dg = np.random.rand(problem.num_random_features)
                hess = oracle.hessian_gamma(beta, gamma)
                maybe_dir = hess.dot(dg)
                true_dir = (oracle.gradient_gamma(beta, gamma + r * dg)
                            - oracle.gradient_gamma(beta, gamma - r * dg)
                            ) / (2 * r)
                self.assertTrue(allclose(maybe_dir, true_dir, rtol=rtol, atol=atol),
                                msg="Hessian gamma does not look right")

                db = np.random.rand(problem.num_fixed_features)
                hess = oracle.hessian_beta_gamma(beta, gamma)
                maybe_dir = hess.T.dot(db)
                true_dir = (oracle.gradient_gamma(beta + r * db, gamma)
                            - oracle.gradient_gamma(beta - r * db, gamma)
                            ) / (2 * r)
                self.assertTrue(allclose(maybe_dir, true_dir, rtol=rtol, atol=atol),
                                msg="Hessian gamma-beta does not look right")

                dg = np.random.rand(problem.num_random_features)
                hess = oracle.hessian_beta_gamma(beta, gamma)
                maybe_dir = hess.dot(dg)
                true_dir = (oracle.gradient_beta(beta, gamma + r * dg)
                            - oracle.gradient_beta(beta, gamma - r * dg)
                            ) / (2 * r)
                self.assertTrue(allclose(maybe_dir, true_dir, rtol=rtol, atol=atol),
                                msg="Hessian beta-gamma does not look right")

    def test_optimal_gamma_consistency_ip_vs_pgd(self):
        trials = 10
        rtol = 1e-2
        atol = 1e-2
        for j in range(trials):
            with self.subTest(j=j):
                problem, true_parameters = LMEProblem.generate(seed=j + 42,
                                                               groups_sizes=[5, 10, 5],
                                                               fit_fixed_intercept=True,
                                                               fit_random_intercept=True,
                                                               features_labels=[FIXED_RANDOM])
                oracle = LinearLMEOracle(problem, n_iter_inner=1000)
                beta = np.random.rand(problem.num_fixed_features)
                gamma = np.random.rand(problem.num_random_features)
                optimal_gamma_pgd = oracle.optimal_gamma(beta, gamma, method="pgd", log_progress=False)
                # pgd_log = np.array(oracle.logger)
                optimal_gamma_ip = oracle.optimal_gamma(beta, gamma, method="ip", log_progress=False)
                # ip_log = np.array(oracle.logger)
                # from matplotlib import pyplot as plt
                # plt.scatter(ip_log[:, 0], ip_log[:, 1], label="ip")
                # plt.scatter(pgd_log[:, 0], pgd_log[:, 1], label="pgd")
                # plt.legend()
                # plt.show()
                self.assertTrue(allclose(optimal_gamma_pgd, optimal_gamma_ip, rtol=rtol, atol=atol),
                                msg="PGD and IP do not match")
                loss_pgd = oracle.loss(beta, optimal_gamma_pgd)
                loss_ip = oracle.loss(beta, optimal_gamma_ip)
                self.assertTrue(allclose(loss_pgd, loss_ip, rtol=rtol, atol=atol))

    def test_no_data_problem(self):
        random_seed = 43
        problem, true_parameters = LMEProblem.generate(groups_sizes=[10, 10, 10],
                                                       features_labels=[],
                                                       fit_fixed_intercept=True,
                                                       fit_random_intercept=True,
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
        problem, true_parameters = LMEProblem.generate(groups_sizes=[4, 5, 10],
                                                       features_labels=[FIXED_RANDOM,
                                                                        FIXED_RANDOM,
                                                                        FIXED,
                                                                        RANDOM],
                                                       fit_fixed_intercept=True,
                                                       fit_random_intercept=False,
                                                       obs_var=0.1,
                                                       seed=42)
        # when both regularization coefficients are zero, these two oracles should be exactly equivalent
        oracle_non_regularized = LinearLMEOracle(problem)
        oracle_regularized = LinearLMEOracleSR3(problem, lg=0, lb=0)
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
            beta1 = oracle_regularized.optimal_beta(random_gamma, tbeta=random_beta)
            beta2 = oracle_non_regularized.optimal_beta(random_gamma)
            self.assertTrue(allclose(beta1, beta2, rtol=rtol, atol=atol),
                            msg="Optimal betas of zero-regularized and non-regularized oracles are different")
            us1 = oracle_regularized.optimal_random_effects(random_beta, random_gamma)
            us2 = oracle_non_regularized.optimal_random_effects(random_beta, random_gamma)
            self.assertTrue(allclose(us1, us2, rtol=rtol, atol=atol),
                            msg="Optimal random effects of zero-regularized and non-regularized oracles is different")
        return None

    def test_beta_to_gamma_map(self):
        problem, true_parameters = LMEProblem.generate(groups_sizes=[4, 5, 10],
                                                       features_labels=[FIXED_RANDOM,
                                                                        FIXED_RANDOM,
                                                                        FIXED,
                                                                        RANDOM,
                                                                        FIXED_RANDOM,
                                                                        FIXED,
                                                                        RANDOM],
                                                       fit_fixed_intercept=True,
                                                       fit_random_intercept=False,
                                                       obs_var=0.1,
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
            problem, true_parameters = LMEProblem.generate(groups_sizes=[40, 30, 50],
                                                           features_labels=[],
                                                           fit_fixed_intercept=True,
                                                           fit_random_intercept=True,
                                                           obs_var=0.1,
                                                           seed=seed)
            oracle = LinearLMEOracle(problem)
            gamma = true_parameters['gamma']
            rho = gamma / (gamma + 0.1)
            oracle._recalculate_cholesky(true_parameters['gamma'])
            n_eff = oracle._jones2010n_eff()
            assert np.allclose(n_eff, sum([ni / (1 + (ni - 1) * rho) for ni in problem.groups_sizes]))

    def test_hodges2001ddf(self):
        # From here:
        # https://www.jstor.org/stable/2673485?seq=1
        problem, true_parameters = LMEProblem.generate(groups_sizes=[40, 30, 50],
                                                       features_labels=[FIXED_RANDOM] * 3,
                                                       fit_random_intercept=True,
                                                       obs_var=0.1,
                                                       seed=42)
        oracle = LinearLMEOracle(problem)
        true_gamma = true_parameters['gamma']
        ddf = oracle._hodges2001ddf(true_gamma)
        #  #|beta| <= DDoF <= #|beta| + num_groups*#|u|
        assert 4 <= ddf <= 4 + 4 * 3

    def test_hat_matrix(self):
        for seed in range(10):
            problem, true_parameters = LMEProblem.generate(groups_sizes=[40, 30, 50],
                                                           features_labels=[FIXED_RANDOM] * 3,
                                                           fit_random_intercept=True,
                                                           obs_var=0.1,
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

    def test_flip_probabilities(self):
        problem, true_parameters = LMEProblem.generate(groups_sizes=[40, 30, 50],
                                                       features_labels=[FIXED_RANDOM] * 2,
                                                       fit_random_intercept=True,
                                                       obs_var=0.1)
        oracle = LinearLMEOracle(problem)
        flip_probabilities = oracle.flip_probabilities_beta(**true_parameters)
        self.assertTrue((0 <= flip_probabilities).all() and (flip_probabilities <= 1).all())


if __name__ == '__main__':
    unittest.main()
