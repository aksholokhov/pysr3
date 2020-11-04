import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import LinearLMESparseModel
from skmixed.lme.problems import LinearLMEProblem

from skmixed.helpers import random_effects_to_matrix


class TestLinearLMESparseModel(unittest.TestCase):

    solvers_to_test = ["pgd",
                       "ip",
                       "ip_combined"]

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
            "nnz_tbeta": 4,
            "nnz_tgamma": 4,
            "lb": 0,        # We expect the coefficient vectors to be dense so we turn regularization off.
            "lg": 0,        # Same.
            "initializer": 'EM',
            "logger_keys": ('converged', 'loss',),
            "tol_inner": 1e-6,
            "tol_outer": 1e-6,
            "n_iter_inner": 1000,
            "n_iter_outer": 1  # we don't care about tbeta and tgamma, so we don't increase regularization iteratively
        }

        max_mse = 0.05
        min_explained_variance = 0.9

        for i in range(trials):
            with self.subTest(i=i):
                problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters, seed=i)
                x, y = problem.to_x_y()
                for solver in self.solvers_to_test:
                    with self.subTest(solver=solver):
                        model = LinearLMESparseModel(solver=solver, **model_parameters)
                        model.fit_problem(problem)
                        logger = model.logger_
                        loss = np.array(logger.get("loss"))
                        if solver == "pgd":
                            self.assertTrue(np.all(loss[1:-1] - loss[:-2] <= 0) and loss[-1] - loss[-2] <= 1e-13,  # sometimes the very last step goes up to machine precision and then stops
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

    def test_solving_sparse_problem(self):
        trials = 10
        problem_parameters = {
            "groups_sizes": [20, 12, 14, 50, 11],
            "features_labels": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "random_intercept": True,
            "obs_std": 0.1,
        }

        model_parameters = {
            "lb": 0,
            "lg": 0,
            "initializer": "None",
            "logger_keys": ('converged', 'loss',),
            "tol_inner": 1e-5,
            "tol_outer": 1e-5,
            "n_iter_inner": 1000,
            "n_iter_outer": 20
        }

        max_mse = 0.12
        min_explained_variance = 0.9
        fixed_effects_min_accuracy = 0.8
        random_effects_min_accuracy = 0.8

        for i in range(trials):
            with self.subTest(i=i):
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

                for solver in self.solvers_to_test:
                    with self.subTest(solver=solver):
                        model = LinearLMESparseModel(**model_parameters,
                                                     solver=solver,
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
        return None

    def test_get_set_params(self):
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
            "initializer": 'EM',
            "logger_keys": ('converged', 'loss',),
            "tol_inner": 1e-4,
            "n_iter_inner": 1,
        }
        # Now we want to solve a regularized problem to get two different models
        model2_parameters = {
            "nnz_tbeta": 3,
            "nnz_tgamma": 2,
            "lb": 20,
            "lg": 20,
            "initializer": None,
            "logger_keys": ('converged',),
            "tol_inner": 1e-4,
            "n_iter_inner": 1,
        }
        problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters, seed=42)
        x, y = problem.to_x_y()

        model = LinearLMESparseModel(**model_parameters)
        # model.fit(x, y)
        model.fit_problem(problem)
        params = model.get_params()
        y_pred = model.predict_problem(problem)

        model2 = LinearLMESparseModel(**model2_parameters)
        #model2.fit(x, y)
        model2.fit_problem(problem)
        params2 = model2.get_params()
        y_pred2 = model2.predict_problem(problem)

        model.set_params(**params2)
        model.fit_problem(problem)
        y_pred_with_other_params = model.predict_problem(problem)
        assert np.equal(y_pred_with_other_params, y_pred2).all(),\
            "set_params or get_params is not working properly"
        model2.set_params(**params)
        model2.fit_problem(problem)
        y_pred2_with_other_params = model2.predict_problem(problem)
        assert np.equal(y_pred2_with_other_params, y_pred).all(), \
            "set_params or get_params is not working properly"

    def test_score_function(self):
        # this is only a basic test which checks R^2 in two points: nearly perfect prediction and constant prediction.

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
            "initializer": 'EM',
            "logger_keys": ('converged', 'loss',),
            "tol_inner": 1e-4,
            "n_iter_inner": 1,
        }

        problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters, seed=42)
        x, y = problem.to_x_y()
        model = LinearLMESparseModel(**model_parameters)
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
