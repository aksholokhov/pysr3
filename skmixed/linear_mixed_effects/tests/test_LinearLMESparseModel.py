import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.linear_mixed_effects.models import LinearLMESparseModel
from skmixed.linear_mixed_effects.problems import LinearLMEProblem


class TestLinearLMESparseModel(unittest.TestCase):

    def test_solving_dense_problem(self):
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
            "nnz_tbeta": 2,
            "nnz_tgamma": 2,
            "lb": 0,        # We expect the coefficient vectors to be dense so we turn regularization off.
            "lg": 0,        # Same.
            "initializer": 'EM',
            "logger_keys": ('converged', 'loss',),
            "tol": 1e-6,
            "n_iter": 1000,
            "tol_inner": 1e-4,
            "n_iter_inner": 1000,
        }

        max_mse = 0.05
        min_explained_variance = 0.9

        for i in range(trials):
            problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters, seed=i)
            model = LinearLMESparseModel(**model_parameters)

            # per_group_coefficients = true_model_parameters["per_group_coefficients"]
            # us = true_model_parameters['random_effects']
            # empirical_gamma = np.sum(us ** 2, axis=0) / problem.num_studies

            x, y = problem.to_x_y()
            model.fit(x, y)

            logger = model.logger_
            loss = np.array(logger.get("loss"))
            self.assertTrue(np.all(loss[1:] - loss[:-1] <= 0),
                            msg="%d) Loss does not decrease monotonically with iterations. (seed=%d)" % (i, i))

            y_pred = model.predict(x)
            explained_variance = explained_variance_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            # coefficients = model.coef_
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

            # coefficients = model.coef_
            # maybe_per_group_coefficients = coefficients["per_group_coefficients"]
            # maybe_beta = coefficients["beta"]
            # maybe_us = coefficients["random_effects"]
            # maybe_gamma = coefficients["gamma"]
            # maybe_tbeta = coefficients["tbeta"]
            # maybe_tgamma = coefficients["tgamma"]
            # maybe_cluster_coefficients = coefficients["per_cluster_coefficients"]
            # maybe_sparse_cluster_coefficients = coefficients["sparse_per_cluster_coefficients"]
        # cluster_coefficients = beta + us
        # maybe_cluster_coefficients = maybe_beta + maybe_us
        return None

    def test_solving_sparse_problem(self):
        trials = 10
        problem_parameters = {
            "groups_sizes": [20, 12, 14, 50, 11],
            "features_labels": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "random_intercept": True,
            "obs_std": 0.1,
        }

        true_beta = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        true_gamma = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

        model_parameters = {
            # "nnz_tbeta": 4,
            # "nnz_tgamma": 3,
            "lb": 20,
            "lg": 2,
            "initializer": None,
            "logger_keys": ('converged', 'loss',),
            "tol": 1e-6,
            "n_iter": 1000,
            "tol_inner": 1e-4,
            "n_iter_inner": 1000,
        }

        max_mse = 0.05
        min_explained_variance = 0.9
        fixed_effects_min_accuracy = 0.7
        random_effects_min_accuracy = 0.7

        for i in range(trials):
            np.random.seed(i)
            true_beta = np.random.choice(2, size=11, p=np.array([0.5, 0.5]))
            if sum(true_beta) == 0:
                true_beta[0] = 1
            true_gamma = np.random.choice(2, size=11, p=np.array([0.3, 0.7]))*true_beta

            problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                       beta=true_beta,
                                                                       gamma=true_gamma,
                                                                       seed=i)
            model = LinearLMESparseModel(**model_parameters,
                                         nnz_tbeta=sum(true_beta),
                                         nnz_tgamma=sum(true_gamma))

            # per_group_coefficients = true_model_parameters["per_group_coefficients"]
            # us = true_model_parameters['random_effects']
            # empirical_gamma = np.sum(us ** 2, axis=0) / problem.num_studies

            x, y = problem.to_x_y()
            model.fit(x, y)

            logger = model.logger_
            loss = np.array(logger.get("loss"))
            self.assertTrue(np.all(loss[1:] - loss[:-1] <= 0),
                            msg="%d) Loss does not decrease monotonically with iterations. (seed=%d)" % (i, i))

            y_pred = model.predict(x)
            explained_variance = explained_variance_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            coefficients = model.coef_
            maybe_beta = coefficients["beta"]
            maybe_gamma = coefficients["gamma"]
            maybe_tbeta = coefficients["tbeta"]
            maybe_tgamma = coefficients["tgamma"]
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



            # coefficients = model.coef_
            # maybe_per_group_coefficients = coefficients["per_group_coefficients"]
            # maybe_beta = coefficients["beta"]
            # maybe_us = coefficients["random_effects"]
            # maybe_gamma = coefficients["gamma"]
            # maybe_tbeta = coefficients["tbeta"]
            # maybe_tgamma = coefficients["tgamma"]
            # maybe_cluster_coefficients = coefficients["per_cluster_coefficients"]
            # maybe_sparse_cluster_coefficients = coefficients["sparse_per_cluster_coefficients"]
        # cluster_coefficients = beta + us
        # maybe_cluster_coefficients = maybe_beta + maybe_us
        return None


if __name__ == '__main__':
    unittest.main()
