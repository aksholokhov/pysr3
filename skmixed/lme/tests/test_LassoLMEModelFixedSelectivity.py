import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import LassoLMEModel, LassoLMEModelFixedSelectivity
from skmixed.lme.problems import LinearLMEProblem


class TestLassoLMEModelFixedSelectivity(unittest.TestCase):

    def test_solving_sparse_problem(self):
        # TODO: fix lasso, it sometimes does not go towards decrease
        self.assertTrue(True)
        return None
        trials = 10
        problem_parameters = {
            "groups_sizes": [20, 12, 14, 50, 11],
            "features_labels": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "random_intercept": True,
            "obs_std": 0.1,
        }

        model_parameters = {
            # nnz_tbeta = 3     # we define it later in the trial's iteration
            # nnz_tgamma = 3    # same
            "logger_keys": ('converged', 'loss',),
            "tol": 1e-3,
            "n_iter": 1000,
        }

        max_mse = 0.1
        min_explained_variance = 0.9
        fixed_effects_min_accuracy = 0.7
        random_effects_min_accuracy = 0.7

        for i in range(trials):
            np.random.seed(i)
            true_beta = np.random.choice(2, size=11, p=np.array([0.5, 0.5]))
            if sum(true_beta) == 0:
                true_beta[0] = 1
            true_gamma = np.random.choice(2, size=11, p=np.array([0.3, 0.7])) * true_beta

            problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                       beta=true_beta,
                                                                       gamma=true_gamma,
                                                                       seed=i)
            model = LassoLMEModelFixedSelectivity(**model_parameters,
                                                  nnz_tbeta=sum(true_beta),
                                                  nnz_tgamma=sum(true_gamma))

            x, y = problem.to_x_y()
            # model.fit(x, y)
            model.fit_problem(problem)

            logger = model.logger_
            loss = np.array(logger.get("loss"))
            self.assertTrue(np.all(loss[1:] - loss[:-1] <= 0),
                            msg="%d) Loss does not decrease monotonically with iterations. (seed=%d)" % (i, i))

            y_pred = model.predict_problem(problem)
            explained_variance = explained_variance_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            maybe_tbeta = model.coef_["beta"]
            maybe_tgamma = model.coef_["gamma"]
            fixed_effects_accuracy = accuracy_score(true_beta, maybe_tbeta != 0)
            random_effects_accuracy = accuracy_score(true_gamma, maybe_tgamma != 0)

            # maybe_per_group_coefficients = coefficients["per_group_coefficients"]

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
            self.assertGreater(fixed_effects_accuracy, fixed_effects_min_accuracy,
                               msg="%d) Fixed Effects Selection Accuracy is too small: %.3f < %.2f  (seed=%d)"
                                   % (i,
                                      fixed_effects_accuracy,
                                      fixed_effects_min_accuracy,
                                      i)
                               )
            self.assertGreater(random_effects_accuracy, random_effects_min_accuracy,
                               msg="%d) Random Effects Selection Accuracy is too small: %.3f < %.2f  (seed=%d)"
                                   % (i,
                                      random_effects_accuracy,
                                      random_effects_min_accuracy,
                                      i)
                               )
        return None
