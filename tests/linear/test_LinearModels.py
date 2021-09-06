import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score
from sklearn.utils.estimator_checks import check_estimator

from pysr3.linear.models import SimpleLinearModel, SimpleLinearModelSR3, LinearL1Model, LinearL1ModelSR3, \
    LinearCADModel, LinearCADModelSR3, LinearSCADModel, LinearSCADModelSR3
from pysr3.linear.problems import LinearProblem


class TestLinearModels(unittest.TestCase):

    def test_meeting_sklearn_standards(self):
        models_to_test = {
            "Simple": SimpleLinearModel(),
            "L1": LinearL1Model(),
            "CAD": LinearCADModel(),
            "SCAD": LinearSCADModel(),
            "Simple_SR3": SimpleLinearModelSR3(),
            "L1_SR3": LinearL1ModelSR3(),
            "CAD_SR3": LinearCADModelSR3(),
            "SCAD_SR3": LinearSCADModelSR3()
        }
        for name, model in models_to_test.items():
            with self.subTest(name=name):
                check_estimator(model)

    def test_solving_dense_problem(self):

        models_to_test = {
            "Simple": (SimpleLinearModel, {}),
            "L1": (LinearL1Model, {}),
            "CAD": (LinearCADModel, {"rho": 0.5}),
            "SCAD": (LinearSCADModel, {"rho": 3.7, "sigma": 0.5}),
            "Simple_SR3": (SimpleLinearModelSR3, {}),
            "L1_SR3": (LinearL1ModelSR3, {}),
            "CAD_SR3": (LinearCADModelSR3, {"rho": 0.5}),
            "SCAD_SR3": (LinearSCADModelSR3, {"rho": 3.7, "sigma": 0.5})
        }

        trials = 3

        problem_parameters = {
            "num_objects": 100,
            "num_features": 10,
            "obs_std": 0.1,
        }

        default_params = {
            "el": 1,
            "lam": 0.0,  # we expect the answers to be dense so the regularizers are small
            # "stepping": "line-search",
            "logger_keys": ('converged', 'loss',),
            "tol_solver": 1e-6,
            "max_iter_solver": 1000
        }

        max_mse = 0.05
        min_explained_variance = 0.9

        for i in range(trials):
            with self.subTest(i=i):
                for model_name, (model_constructor, local_params) in models_to_test.items():
                    with self.subTest(model_name=model_name):
                        problem = LinearProblem.generate(**problem_parameters, seed=i)
                        x, y = problem.to_x_y()

                        model_params = default_params.copy()
                        model_params.update(local_params)

                        model = model_constructor(**model_params)
                        model.fit_problem(problem)

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
            "L1": (LinearL1Model, {"lam": 2}),
            "CAD": (LinearCADModel, {"rho": 0.5}),
            "SCAD": (LinearSCADModel, {"lam": 1, "rho": 3.7, "sigma": 2.5}),
            "L1_SR3": (LinearL1ModelSR3, {"lam": 0.1}),
            "L1_SR3P": (LinearL1ModelSR3, {"lam": 0.1, "practical": True}),
            "CAD_SR3": (LinearCADModelSR3, {"rho": 0.5}),
            "CAD_SR3P": (LinearCADModelSR3, {"rho": 0.5, "practical": True}),
            "SCAD_SR3": (LinearSCADModelSR3, {"lam": 0.2, "rho": 3.7, "sigma": 0.5}),
            "SCAD_SR3P": (LinearSCADModelSR3, {"lam": 0.2, "rho": 3.7, "sigma": 0.5, "practical": True})
        }
        trials = 5

        problem_parameters = {
            "num_objects": 100,
            "num_features": 20,
            "obs_std": 0.1,
        }

        default_params = {
            "el": 1,
            "lam": 1,
            "rho": 0.3,
            "logger_keys": ('converged', 'loss',),
            "tol_solver": 1e-6,
            "max_iter_solver": 5000
        }

        max_mse = 0.5
        min_explained_variance = 0.9
        min_selection_accuracy = 0.9

        for i in range(trials):
            with self.subTest(i=i):
                for model_name, (model_constructor, local_params) in models_to_test.items():
                    with self.subTest(model_name=model_name):
                        seed = i + 42
                        np.random.seed(seed)
                        true_x = np.random.choice(2, size=problem_parameters["num_features"], p=np.array([0.5, 0.5]))
                        if sum(true_x) == 0:
                            true_x[0] = 1

                        problem = LinearProblem.generate(**problem_parameters,
                                                         true_x=true_x,
                                                         seed=seed)
                        x, y = problem.to_x_y()

                        model_params = default_params.copy()
                        model_params.update(local_params)

                        model = model_constructor(**model_params)
                        model.fit_problem(problem)

                        y_pred = model.predict_problem(problem)
                        explained_variance = explained_variance_score(y, y_pred)
                        mse = mean_squared_error(y, y_pred)

                        coefficients = model.coef_
                        maybe_x = coefficients["x"]
                        selection_accuracy = accuracy_score(true_x, abs(maybe_x) > 1e-2)

                        self.assertGreater(explained_variance, min_explained_variance,
                                           msg=f"{model_name}: Explained variance is too small: {explained_variance} < {min_explained_variance} (seed={seed})")
                        self.assertGreater(max_mse, mse,
                                           msg=f"{model_name}: MSE is too big: {max_mse} > {mse} (seed={seed})")
                        self.assertGreater(selection_accuracy, min_selection_accuracy,
                                           msg=f"{model_name}: Fixed Effects Selection Accuracy is too small: {selection_accuracy} < {min_selection_accuracy}  (seed={seed})")

        return None


if __name__ == '__main__':
    unittest.main()
