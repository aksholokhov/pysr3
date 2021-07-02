import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import Sr3L0LmeModel, L0LmeModel, L1LmeModel, SR3L1LmeModel, CADLmeModel, SR3CADLmeModel
from skmixed.lme.problems import LinearLMEProblem

from skmixed.helpers import random_effects_to_matrix


class TestLmeModels(unittest.TestCase):


    def test_solving_dense_problem(self):

        models_to_test = {
            "L0": L0LmeModel,
            "L1": L1LmeModel,
            "CAD": CADLmeModel,
            "L0SR3": Sr3L0LmeModel,
            "L1SR3": SR3L1LmeModel,
            "CADSR3": SR3CADLmeModel
        }

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
            "nnz_tbeta": 4,
            "nnz_tgamma": 4,
            "lb": 1,
            "lg": 1,
            "rho": 0.1,
            "lam": 0.0, # we expect the answers to be dense so the regularizers are small
            #"stepping": "line-search",
            "initializer": 'None',
            "logger_keys": ('converged', 'loss',),
            "tol_oracle": 1e-3,
            "tol_solver": 1e-6,
            "max_iter_oracle": 1000,
            "max_iter_solver": 1000
        }

        max_mse = 0.05
        min_explained_variance = 0.9

        for i in range(trials):
            with self.subTest(i=i):
                for model_name, model_constructor in models_to_test.items():
                    with self.subTest(model_name=model_name):
                        problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters, seed=i)
                        x, y = problem.to_x_y()

                        model = model_constructor(**model_parameters)
                        model.fit_problem(problem)
                        logger = model.logger_
                        loss = np.array(logger.get("loss"))

                        # self.assertTrue(np.all(loss[1:-1] - loss[:-2] <= 0) and loss[-1] - loss[-2] <= 1e-13,  # sometimes the very last step goes up to machine precision and then stops
                        #                    msg="%d) Loss does not decrease monotonically with iterations. (seed=%d)" % (i, i))

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

    def test_solving_sparse_problem(self):

        models_to_test = {
            "L0": L0LmeModel,
            #"L1": L1LmeModel,
            "L0SR3": Sr3L0LmeModel,
            "L1SR3": SR3L1LmeModel,
            "CADSR3": SR3CADLmeModel,
        }

        trials = 10

        problem_parameters = {
            "groups_sizes": [20, 12, 14, 50, 11],
            "features_labels": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "random_intercept": True,
            "obs_std": 0.1,
        }

        model_parameters = {
            "lb": 20,
            "lg": 20,
            "initializer": "EM",
            "lam": 1,
            "rho": 0.1,
            "logger_keys": ('converged', 'loss',),
            "tol_oracle": 1e-3,
            "tol_solver": 1e-6,
            "max_iter_oracle": 1000,
            "max_iter_solver": 1000
        }

        max_mse = 0.12
        min_explained_variance = 0.9
        fixed_effects_min_accuracy = 0.8
        random_effects_min_accuracy = 0.8

        for i in range(trials):
            with self.subTest(i=i):
                for model_name, model_constructor in models_to_test.items():
                    with self.subTest(model_name=model_name):
                        np.random.seed(i)
                        true_beta = np.random.choice(2, size=11, p=np.array([0.5, 0.5]))
                        if sum(true_beta) == 0:
                            true_beta[0] = 1
                        true_gamma = np.random.choice(2, size=11, p=np.array([0.3, 0.7])) * true_beta

                        problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                                   beta=true_beta,
                                                                                   gamma=true_gamma,
                                                                                   seed=i)
                        x, y = problem.to_x_y()


                        model = model_constructor(**model_parameters,
                                                 nnz_tbeta=sum(true_beta),
                                                 nnz_tgamma=sum(true_gamma))
                        model.fit_problem(problem)

                        logger = model.logger_
                        loss = np.array(logger.get("loss"))
                        # TODO: It won't decrease monotonically because it may jump when we increase regularization.
                        # self.assertTrue(np.all(loss[1:] - loss[:-1] <= 0),
                        #                 msg="%d) Loss does not decrease monotonically with iterations. (seed=%d)" % (i, i))

                        y_pred = model.predict_problem(problem)
                        explained_variance = explained_variance_score(y, y_pred)
                        mse = mean_squared_error(y, y_pred)

                        coefficients = model.coef_
                        maybe_tbeta = coefficients["beta"]
                        maybe_tgamma = coefficients["gamma"]
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


    def test_score_function(self):
        # this is only a basic test which checks R^2 in two points: nearly perfect prediction and constant prediction.
        models_to_test = {
            "L0": L0LmeModel,
            "L1": L1LmeModel,
            "L0SR3": Sr3L0LmeModel,
            "L1SR3": SR3L1LmeModel,
        }
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
            "nnz_tbeta": 4,
            "nnz_tgamma": 4,
            "lb": 0,  # We expect the coefficient vectors to be dense so we turn regularization off.
            "lg": 0,  # Same.
            "lam": 10,
            "initializer": 'EM',
            "logger_keys": ('converged', 'loss',),
            "tol_oracle": 1e-6,
            "tol_solver": 1e-6,
            "max_iter_oracle": 1,
            "max_iter_solver": 1
            # we don't care about tbeta and tgamma, so we don't increase regularization iteratively
        }

        problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters, seed=42)
        x, y = problem.to_x_y()
        for model_name, model_constructor in models_to_test.items():
            with self.subTest(model_name=model_name):
                model = model_constructor(**model_parameters)
                model.fit_problem(problem)
                model.coef_["beta"] = true_model_parameters["beta"]
                model.coef_["random_effects"] = random_effects_to_matrix(true_model_parameters["random_effects"])
                good_score = model.score(x, y)
                assert good_score > 0.99
                model.coef_["beta"] = np.zeros(4)
                model.coef_["random_effects"] = np.zeros((4, 4))
                bad_score = model.score(x, y)
                assert abs(bad_score) < 0.1


if __name__ == '__main__':
    unittest.main()
