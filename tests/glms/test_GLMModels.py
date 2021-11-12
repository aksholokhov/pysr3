import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score
from sklearn.utils.estimator_checks import check_estimator

from pysr3.glms.models import SimplePoissonModel, PoissonL1Model, PoissonL1ModelSR3
from pysr3.glms.problems import PoissonProblem


class TestGLMs(unittest.TestCase):

    def test_meeting_sklearn_standards(self):
        models_to_test = {
            "PoissonSimple": SimplePoissonModel(),
            "PoissonL1": PoissonL1Model(),
            "PoissonL1SR3": PoissonL1ModelSR3(practical=True)

        }
        for name, model in models_to_test.items():
            with self.subTest(name=name):
                check_estimator(model)

    def test_solving_dense_problem(self):
        trials = 3

        problem_parameters = {
            "num_objects": 100,
            "num_features": 10,
            "true_x": np.ones(10) / 10,
            "obs_std": 0.1,
        }

        models_to_test = {
            "PoissonSimple": (SimplePoissonModel, {}),
            "PoissonL1": (PoissonL1Model, {}),
            "PoissonL1SR3Constrained": (
                PoissonL1ModelSR3, {"practical": True,
                                    "constraints": ([-3] * (problem_parameters["num_features"] + 1),
                                                    [3] * (problem_parameters["num_features"] + 1))}),
            "PoissonL1SR3NonConstrained": (PoissonL1ModelSR3, {})
        }

        default_params = {
            "el": 1,
            "alpha": 0.0,  # we expect the answers to be dense so the regularizers are small
            # "stepping": "line-search",
            "logger_keys": ('converged', 'loss',),
            "tol_solver": 1e-6,
            "max_iter_solver": 1000
        }

        max_rmse = 1
        min_explained_variance = 0.9

        for i in range(trials):
            with self.subTest(i=i):
                for model_name, (model_constructor, local_params) in models_to_test.items():
                    with self.subTest(model_name=model_name):
                        problem = PoissonProblem.generate(**problem_parameters, seed=i)
                        _, y = problem.to_x_y()

                        model_params = default_params.copy()
                        model_params.update(local_params)

                        model = model_constructor(**model_params)
                        model.fit_problem(problem)

                        y_pred = model.predict_problem(problem)
                        # explained_variance = explained_variance_score(y, y_pred)
                        rmse = np.mean(((y - y_pred) / (y_pred + 1)) ** 2)

                        # self.assertGreater(explained_variance, min_explained_variance,
                        #                    msg="%d) Explained variance is too small: %.3f < %.3f. (seed=%d)"
                        #                        % (i,
                        #                           explained_variance,
                        #                           min_explained_variance,
                        #                           i))
                        self.assertGreater(max_rmse, rmse,
                                           msg="%d) RMSE is too big: %.3f > %.2f  (seed=%d)"
                                               % (i,
                                                  rmse,
                                                  max_rmse,
                                                  i))
        return None

    def test_solving_sparse_problem(self):

        problem_parameters = {
            "num_objects": 500,
            "num_features": 20,
            "obs_std": 0.1,
        }

        models_to_test = {
            "PoissonL1": (PoissonL1Model, {"alpha": 5}),
            "PoissonL1SR3": (PoissonL1ModelSR3, {
                "fit_intercept": True,
                "practical": True,
                "constraints": ([-5] * (problem_parameters["num_features"]+1),
                                                                 [5] * (problem_parameters["num_features"]+1))})
        }
        trials = 5

        default_params = {
            "el": 1,
            "alpha": 0.5,
            "rho": 0.3,
            "logger_keys": ('converged', ),
            "tol_solver": 1e-8,
            "max_iter_solver": 5000
        }

        max_rmse = 1
        min_explained_variance = 0.9
        min_selection_accuracy = 0.9

        for i in range(trials):
            with self.subTest(i=i):
                for model_name, (model_constructor, local_params) in models_to_test.items():
                    with self.subTest(model_name=model_name):
                        seed = i
                        np.random.seed(seed)
                        true_x = np.random.choice(2, size=problem_parameters["num_features"], p=np.array([0.5, 0.5]))
                        if sum(true_x) == 0:
                            true_x[0] = 1
                        true_x = 5 * true_x / sum(true_x)

                        problem = PoissonProblem.generate(**problem_parameters,
                                                          true_x=true_x,
                                                          seed=seed)
                        x, y = problem.to_x_y()

                        model_params = default_params.copy()
                        model_params.update(local_params)

                        model = model_constructor(**model_params)
                        model.fit_problem(problem)

                        y_pred = model.predict_problem(problem)
                        # probably not the best score for a discrete distribution
                        explained_variance = explained_variance_score(y, y_pred)
                        rmse = np.mean(((y - y_pred) / (y_pred + 1)) ** 2)
                        iters = model.logger_.get("iteration")
                        maybe_x = model.coef_  # intercept was not used for problem generation
                        selection_accuracy = accuracy_score(abs(true_x) > 1e-2, abs(maybe_x) > 1e-2)

                        # self.assertGreaterEqual(explained_variance, min_explained_variance,
                        #                    msg=f"{model_name}: Explained variance is too small: {explained_variance} < {min_explained_variance} (seed={seed})")
                        self.assertGreaterEqual(max_rmse, rmse,
                                                msg=f"{model_name}: MSE is too big: {rmse} > {max_rmse} (seed={seed})")
                        self.assertGreaterEqual(selection_accuracy, min_selection_accuracy,
                                                msg=f"{model_name}: Feature Selection Accuracy is too small: {selection_accuracy} < {min_selection_accuracy}  (seed={seed})")

        return None