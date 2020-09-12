import unittest
from unittest import TestCase

import numpy as np
from numpy import allclose
from scipy.misc import derivative

from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracleW, LinearLMEOracle


class TestLinearLMEOracleW(TestCase):

    def test_drop_matrices(self):

        problem_parameters = {
            "groups_sizes": [20, 5, 10, 50],
            "features_labels": [1, 2, 3, 3],
            "random_intercept": True,
            "obs_std": 0.1,
            "seed": 42
        }

        problem, _ = LinearLMEProblem.generate(**problem_parameters)
        simple_oracle = LinearLMEOracle(problem)
        oracle = LinearLMEOracleW(problem, lb=0, lg=0,
                                  nnz_tbeta=problem.num_fixed_effects,
                                  nnz_tgamma=problem.num_random_effects)
        trials = 100

        rtol = 1e-10
        atol = 1e-10
        np.random.seed(42)

        for t, (random_beta, random_gamma) in enumerate(zip(np.random.rand(trials, problem.num_fixed_effects),
                                                            np.random.rand(trials, problem.num_random_effects))):
            loss = simple_oracle.loss(random_beta, random_gamma)
            oracle._recalculate_drop_matrices(random_beta, random_gamma)
            w_beta = oracle.drop_penalties_beta
            w_gamma = oracle.drop_penalties_gamma
            for j in range(problem.num_fixed_effects):
                sparse_beta = random_beta.copy()
                sparse_beta[j] = 0
                sparse_gamma = random_gamma.copy()
                idx = oracle.beta_to_gamma_map[j].astype(int)
                if idx >= 0:
                    sparse_gamma[idx] = 0
                    loss3 = simple_oracle.loss(random_beta, sparse_gamma)
                    self.assertTrue(np.isclose(loss3 - loss, w_gamma[idx], rtol=rtol, atol=atol),
                                    msg="%d: W_gamma is not right" % j)
                    loss2 = simple_oracle.loss(sparse_beta, sparse_gamma)
                else:
                    loss2 = simple_oracle.loss(sparse_beta, random_gamma)
                self.assertTrue(np.isclose(loss2 - loss, w_beta[j], rtol=rtol, atol=atol),
                                msg="%d) W_beta is not right" % j)

        sparse_beta = np.zeros(problem.num_fixed_effects)
        sparse_gamma = np.zeros(problem.num_random_effects)
        sparse_beta[0:2] = 1
        sparse_gamma[0] = 1
        oracle._recalculate_drop_matrices(sparse_beta, sparse_gamma)
        w_beta = oracle.drop_penalties_beta
        w_gamma = oracle.drop_penalties_gamma
        self.assertTrue((w_gamma[1:] == 0).all(), msg="Drop of zero gamma is not zero")
        self.assertTrue((w_beta[2:] == 0).all(), msg="Drop of zero beta is not zero")
