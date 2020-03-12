import unittest
from unittest import TestCase

import numpy as np
from numpy import allclose

from linear_mixed_effects.problems import LinearLMEProblem
from skmixed.legacy.oracles import LinearLMEOracle
from skmixed.legacy.solvers import LinearLMESolver


def test_convergence_for_method(method="EM", trials=100, noise_variance=1e-2, loss_tol=1e-4, max_iter=1000, rtol=2e-1, atol=1e-1):
    converged = 0
    beta_close = []
    gamma_close = []
    loss_decreases = []
    for random_seed in np.random.randint(0, 1000, size=trials):
        problem, true_beta, true_gamma, true_us, cov_mats = LinearLMEProblem.generate(obs_std=noise_variance,
                                                                                      seed=random_seed)
        oracle = LinearLMEOracle(problem, mode='fast')
        alg = LinearLMESolver(tol=loss_tol, max_iter=max_iter)

        try:
            logger = alg.fit(oracle, beta0=0.85 * true_beta, gamma0=0.85 * true_gamma,
                                                         method=method)
        except np.linalg.LinAlgError as err:
            continue

        pred_beta = alg.beta
        pred_gamma = alg.gamma

        if logger['converged'] is 0:
            continue

        converged += 1
        beta_close.append(allclose(pred_beta, true_beta, rtol=rtol, atol=atol))
        gamma_close.append(allclose(pred_gamma, true_gamma, rtol=rtol, atol=atol))
        loss_decreases.append((np.array(logger["loss"])[1:] - np.array(logger["loss"])[:-1] < 0).all())

    return converged/trials, beta_close, gamma_close, loss_decreases



class TestLinearLMESolver(TestCase):

    def test_convergence_em(self):
        pass
        converged, beta_close, gamma_close, loss_decreases = test_convergence_for_method(method="EM")
        #self.assertGreater(converged, 0.9)
        self.assertGreater(np.mean(beta_close), 0.8)
        #self.assertGreater(np.mean(gamma_close), 0.5)
        #self.assertGreater(np.mean(loss_decreases), 0.8)

    def test_convergence_nr(self):
        converged, beta_close, gamma_close, loss_decreases = test_convergence_for_method(
            method="NewtonRaphson", max_iter=50
        )
        # self.assertGreater(converged, 0.9)
        self.assertGreater(np.mean(beta_close), 0.8)
        # self.assertGreater(np.mean(gamma_close), 0.5)
        #self.assertGreater(np.mean(loss_decreases), 0.8)

    def test_convergence_gd(self):
        pass
        converged, beta_close, gamma_close, loss_decreases = test_convergence_for_method(
            method="GradDescent", max_iter=200)
        #self.assertGreater(converged, 0.9)
        self.assertGreater(np.mean(beta_close), 0.8)
        #self.assertGreater(np.mean(gamma_close), 0.5)
        #self.assertGreater(np.mean(loss_decreases), 0.8)


if __name__ == '__main__':
    unittest.main()
