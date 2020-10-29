import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import LassoLMEModel
from skmixed.lme.problems import LinearLMEProblem


class TestLassoLMEModel(unittest.TestCase):

    def test_solving_lasso(self):
        trials = 20
        problem_parameters = {
            "groups_sizes": [20, 5, 10, 50],
            "features_labels": [3, 3, 3],
            "random_intercept": True,
            "features_covariance_matrix": np.array([
                [1, 0, 0],
                [0, 1, 0.7],
                [0, 0.7, 1]
            ]),
            "obs_std": 0.1,
        }
        model_parameters = {
            "lb": 1,
            "lg": 1,
            "logger_keys": ('converged', 'loss',),
            "tol": 1e-3,
            "n_iter": 1000,
        }

        max_mse = 0.05
        min_explained_variance = 0.9

        for i in range(trials):
            with self.subTest(i=i):
                problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters, seed=i)
                x, y = problem.to_x_y()

                model = LassoLMEModel(**model_parameters)
                model.fit_problem(problem)

                logger = model.logger_
                loss = np.array(logger.get("loss"))
                self.assertTrue(np.all(loss[1:] - loss[:-1] <= 0),
                                msg="%d) Loss does not decrease monotonically with iterations. (seed=%d)" % (i, i))

                y_pred = model.predict_problem(problem)
                explained_variance = explained_variance_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)

                self.assertGreater(explained_variance, min_explained_variance,
                                   msg="%d) Explained variance is too small: %.3f < %.3f. (seed=%d)"
                                       % (i,
                                          explained_variance,
                                          min_explained_variance,
                                          i))
                self.assertGreater(max_mse, mse,
                                   msg="%d) MSE is too big: %.3f > %.2f  (seed=%d)"
                                       % (i,
                                          mse,
                                          max_mse,
                                          i))

        return None
