import unittest
from unittest import TestCase

import numpy as np
from scipy.misc import derivative

from problems import LinearLMEProblem
from legacy.old_solvers import LinearLMESolver


class TestLinearLMESolver(TestCase):

    def test_gamma_derivatives(self):
        trials = 5
        for random_seed in np.random.randint(0, 1000, size=trials):
            np.random.seed(random_seed)
            problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(num_random_effects=2,
                                                                                       seed=random_seed)
            alg = LinearLMESolver()
            alg.fit(problem, no_calculations=True)
            A = np.random.rand(100, 2)
            dx = 1e-7
            beta = np.random.rand(len(beta))
            C = np.array([alg.grad_loss_gamma(beta, g) for g in A])
            D = np.array([derivative(lambda x: alg.loss(beta, np.array([x, g[1]])), g[0], dx=dx) for g in A])
            E = np.array([derivative(lambda x: alg.loss(beta, np.array([g[0], x])), g[1], dx=dx) for g in A])
            for a, c, d, e in zip(A, C, D, E):
                self.assertAlmostEqual(c[0], d, delta=2 * np.sqrt(dx))
                self.assertAlmostEqual(c[1], e, delta=2 * np.sqrt(dx))
        return None

    def test_fast_gamma_derivatives(self):
        trials = 100
        random_seed = 35
        problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(seed=random_seed)
        fast_alg = LinearLMESolver(mode='fast')
        slow_alg = LinearLMESolver(mode='naive')
        fast_alg.fit(problem, no_calculations=True)
        slow_alg.fit(problem, no_calculations=True)
        for i in range(trials):
            beta = np.random.rand(problem.num_features)
            gamma = np.random.rand(problem.num_random_effects)
            true_grad_gamma = slow_alg.grad_loss_gamma(beta, gamma)
            maybe_grad_gamma = fast_alg.grad_loss_gamma(beta, gamma)
            for true_g, pred_g in zip(true_grad_gamma, maybe_grad_gamma):
                self.assertAlmostEqual(true_g, pred_g, delta=1e-8)

    def test_hessian_gamma(self):
        trials = 100
        random_seed = 34
        problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(seed=random_seed)
        fast_alg = LinearLMESolver(mode='fast')
        slow_alg = LinearLMESolver(mode='naive')
        fast_alg.fit(problem, no_calculations=True)
        slow_alg.fit(problem, no_calculations=True)
        np.random.seed(random_seed)
        for j in range(trials):
            beta = np.random.rand(problem.num_features) + 0.1
            gamma = np.random.rand(problem.num_random_effects) + 0.1
            r = 0.00001
            dg = np.random.rand(problem.num_random_effects)
            hess = fast_alg.hessian_gamma(beta, gamma)
            maybe_dir = hess.dot(dg)
            true_dir = (slow_alg.grad_loss_gamma(beta, gamma + r * dg) - slow_alg.grad_loss_gamma(beta,
                                                                                                  gamma - r * dg)) / (
                               2 * r)
            err = np.linalg.norm(maybe_dir - true_dir)
            self.assertAlmostEqual(err, 0, delta=10 * r)

    # TODO: Fix beta and us instability tests
    def test_optimal_beta(self):
        trials = 100
        for random_seed in np.random.randint(0, 1000, size=trials):
            np.random.seed(random_seed)
            noise_variance = 1e-2
            # This test is unstable when randomize more parameters #TODO: fix it
            problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(groups_sizes=[20, 30, 50],
                                                                                       obs_std=noise_variance,
                                                                                       num_fixed_effects=3,
                                                                                       gamma=np.array([1, 1]),
                                                                                       seed=random_seed)
            alg = LinearLMESolver(mode='naive')
            alg.fit(problem, no_calculations=True)
            maybe_beta = alg.optimal_beta(gamma)
            for b, mb in zip(beta, maybe_beta):
                self.assertAlmostEqual(b, mb, delta=5 * noise_variance, msg="b=%f, mb=%f" % (b, mb))

    def test_fast_optimal_beta(self):
        trials = 100
        random_seed = 35
        problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(seed=random_seed)
        fast_alg = LinearLMESolver(mode='fast')
        slow_alg = LinearLMESolver(mode='naive')
        fast_alg.fit(problem, no_calculations=True)
        slow_alg.fit(problem, no_calculations=True)
        for i in range(trials):
            beta = np.random.rand(problem.num_features)
            gamma = np.random.rand(problem.num_random_effects)
            true_beta = slow_alg.optimal_beta(gamma)
            _ = fast_alg.grad_loss_gamma(beta, gamma)
            maybe_beta = fast_alg.optimal_beta(gamma)
            self.assertAlmostEqual(np.linalg.norm(true_beta - maybe_beta), 0)

    def test_optimal_random_effects(self):
        trials = 100
        noise_variance = 1e-2
        for random_seed in np.random.randint(0, 1000, size=trials):
            np.random.seed(random_seed)

            problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(groups_sizes=[20, 30, 50],
                                                                                       obs_std=noise_variance,
                                                                                       num_fixed_effects=3,
                                                                                       gamma=np.array([1, 1]),
                                                                                       seed=random_seed)
            alg = LinearLMESolver()
            alg.fit(problem, no_calculations=True)
            maybe_random_effects = alg.optimal_random_effects(beta, gamma)
            for rnd, mb_rnd in zip(random_effects, maybe_random_effects):
                for u, mb_u in zip(rnd, mb_rnd):
                    self.assertAlmostEqual(u, mb_u, delta=5 * noise_variance,
                                           msg="u=%f, mb_u=%f" % (u, mb_u))

    def test_convergence(self):
        trials = 100
        noise_variance = 1e-2
        loss_tol = 1e-3
        max_iter = 10000
        for random_seed in np.random.randint(0, 1000, size=trials):
            problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(obs_std=noise_variance,
                                                                                       seed=random_seed)
            alg = LinearLMESolver(tol=loss_tol, max_iter=max_iter, mode='naive')
            try:
                logger = alg.fit(problem)
                self.assertEqual(logger['converged'], 1, msg="seed = %d" % random_seed)
            except np.linalg.LinAlgError as err:
                self.assertTrue(False, msg="Seed %d: %s" % (random_seed, err))

    def test_convergence_fast(self):
        trials = 100
        noise_variance = 1e-2
        loss_tol = 1e-3
        max_iter = 10000
        for random_seed in np.random.randint(0, 1000, size=trials):
            problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(obs_std=noise_variance,
                                                                                       seed=random_seed)
            alg = LinearLMESolver(tol=loss_tol, max_iter=max_iter, mode='fast')
            try:
                logger = alg.fit(problem)
                self.assertEqual(logger['converged'], 1, msg="seed = %d" % random_seed)
            except np.linalg.LinAlgError as err:
                self.assertTrue(False, msg="Seed %d: %s" % (random_seed, err))


    # TODO: Fix this convergence test
    def test_convergence_empirical(self):
        trials = 100
        noise_variance = 1e-2
        loss_tol = 1e-3
        max_iter = 10000
        for random_seed in np.random.randint(0, 1000, size=trials):
            problem, beta, gamma, random_effects, cov_mats = LinearLMEProblem.generate(obs_std=noise_variance,
                                                                                       seed=random_seed)
            alg = LinearLMESolver(tol=loss_tol, max_iter=max_iter, mode='fast', method="empirical")
            try:
                logger = alg.fit(problem)
                self.assertEqual(logger['converged'], 1, msg="seed = %d" % random_seed)
            except np.linalg.LinAlgError as err:
                self.assertTrue(False, msg="Seed %d: %s" % (random_seed, err))

if __name__ == '__main__':
    unittest.main()
