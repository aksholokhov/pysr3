import unittest

import numpy as np

from lib.linear_models import LinearLMESparseModel
from lib.problems import LinearLMEProblem


class TestLinearLMESparseModel(unittest.TestCase):
    def test_loss_decreases_over_iterations(self):
        problem_parameters = {
            "study_sizes": [20, 5, 10, 50],
            "num_fixed_effects": 3,
            "num_random_effects": 2,
            "both_fixed_and_random_effects": np.array([0, 1]),
            "features_covariance_matrix": np.array([
                [1, 0, 0],
                [0, 1, 0.7],
                [0, 0.7, 1]
            ]),
            "random_features_covariance_matrix": np.array([
                [1, 0.8],
                [0.8, 1]
            ]),
            "obs_std": 0.1,
            "seed": 55
        }
        model_parameters = {
            "nnz_tbeta": 3,
            "nnz_tgamma": 2,
            "lb": 0,
            "lg": 0,
            "initializer": 'EM',
            "logger_keys": ('converged', 'loss', )
        }
        problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters)
        us = true_model_parameters['random_effects']
        empirical_gamma = np.sum(us ** 2, axis=0) / problem.num_studies
        x, y = problem.to_x_y()
        model = LinearLMESparseModel(**model_parameters)
        model.fit(x, y, random_intercept=False)
        logger = model.logger_
        loss = np.array(logger.get("loss"))
        self.assertTrue(np.all(loss[1:] - loss[:-1] <= 0), msg="Loss does not decrease monotonically with iterations.")
        coefficients = model.coef_
        beta = true_model_parameters["beta"]
        us = true_model_parameters["random_effects"]
        maybe_beta = coefficients["beta"]
        maybe_us = coefficients["random_effects"]
        maybe_gamma = coefficients["gamma"]
        #cluster_coefficients = beta + us
        #maybe_cluster_coefficients = maybe_beta + maybe_us
        pass

if __name__ == '__main__':
    unittest.main()
