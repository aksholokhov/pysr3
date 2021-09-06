import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import loguniform

from pysr3.lme.models import L0LmeModelSR3, L0LmeModel, L1LmeModel, L1LmeModelSR3, CADLmeModel, CADLmeModelSR3, \
    SCADLmeModel, SCADLmeModelSR3, SimpleLMEModel, SimpleLMEModelSR3
from pysr3.lme.problems import LMEProblem, LMEStratifiedShuffleSplit, FIXED_RANDOM
from pysr3.lme.problems import random_effects_to_matrix


class TestLmeModels(unittest.TestCase):

    def test_meeting_sklearn_standards(self):
        models_to_test = {
            "Simple": SimpleLMEModel(),
            "Simple_SR3": SimpleLMEModelSR3(),
            "L0": L0LmeModel(),
            "L1": L1LmeModel(),
            "CAD": CADLmeModel(),
            "SCAD": SCADLmeModel(),
            "L0_SR3": L0LmeModelSR3(),
            "L1_SR3": L1LmeModelSR3(),
            "CAD_SR3": CADLmeModelSR3(),
            "SCAD_SR3": SCADLmeModelSR3()
        }

        for name, model in models_to_test.items():
            with self.subTest(name=name):
                check_estimator(model)

    def test_solving_dense_problem(self):

        models_to_test = {
            "L0": (L0LmeModel, {"stepping": "line-search"}),
            "L1": (L1LmeModel, {"stepping": "line-search"}),
            "CAD": (CADLmeModel, {"stepping": "line-search"}),
            "SCAD": (SCADLmeModel, {"stepping": "line-search", "rho": 3.7, "sigma": 2.5}),
            "L0SR3": (L0LmeModelSR3, {}),
            "L1SR3": (L1LmeModelSR3, {}),
            "CADSR3": (CADLmeModelSR3, {}),
            "SCADSR3": (SCADLmeModelSR3, {"rho": 3.7, "sigma": 2.5})
        }

        trials = 3

        problem_parameters = {
            "groups_sizes": [20, 15, 10, 50],
            "features_labels": [FIXED_RANDOM] * 3,
            "fit_fixed_intercept": True,
            "fit_random_intercept": True,
            "features_covariance_matrix": np.array([
                [1, 0, 0],
                [0, 1, 0.7],
                [0, 0.7, 1]
            ]),
            "obs_var": 0.1,
        }
        default_params = {
            "nnz_tbeta": 4,
            "nnz_tgamma": 4,
            "ell": 1,
            "rho": 0.1,
            "lam": 0.0,  # we expect the answers to be dense so the regularizers are small
            # "stepping": "line-search",
            "initializer": 'None',
            "logger_keys": ('converged', 'loss',),
            "tol_oracle": 1e-4,
            "tol_solver": 1e-6,
            "max_iter_oracle": 1000,
            "max_iter_solver": 5000
        }

        max_mse = 0.1
        min_explained_variance = 0.9

        for i in range(trials):
            with self.subTest(i=i):
                for model_name, (model_constructor, local_params) in models_to_test.items():
                    with self.subTest(model_name=model_name):
                        problem, true_model_parameters = LMEProblem.generate(**problem_parameters, seed=i)
                        x, y, _ = problem.to_x_y()

                        model_params = default_params.copy()
                        model_params.update(local_params)

                        model = model_constructor(**model_params)
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
            "L0": (L0LmeModel, {"stepping": "line-search"}),
            "L1": (L1LmeModel, {"stepping": "line-search"}),
            "CAD": (CADLmeModel, {"rho": 0.3, "stepping": "line-search"}),
            "SCAD": (SCADLmeModel, {"rho": 3.7, "lam": 10, "stepping": "line-search"}),
            "L0_SR3": (L0LmeModelSR3, {}),
            "L1_SR3": (L1LmeModelSR3, {}),
            "CAD_SR3": (CADLmeModelSR3, {"rho": 0.3}),
            "SCAD_SR3": (SCADLmeModelSR3, {"rho": 3.7})
        }

        trials = 2
        n_features = 20
        problem_parameters = {
            "groups_sizes": [30] * 6,
            "features_labels": [FIXED_RANDOM] * n_features,
            "obs_var": 0.01,
        }

        default_params = {
            "ell": 80,
            "initializer": "EM",
            "lam": 5,
            "rho": 0.3,
            "sigma": 0.5,
            # "stepping": "line-search",
            "logger_keys": ('converged', 'loss', 'vaida_aic', 'jones_bic', 'muller_ic'),
            "tol_oracle": 1e-4,
            "tol_solver": 1e-5,
            "max_iter_oracle": 1000,
            "max_iter_solver": 5000
        }

        max_mse = 0.2
        min_explained_variance = 0.9
        fixed_effects_min_accuracy = 0.7
        random_effects_min_accuracy = 0.7

        for i in range(trials):
            with self.subTest(i=i):
                for model_name, (model_constructor, local_params) in models_to_test.items():
                    with self.subTest(model_name=model_name):

                        seed = i
                        np.random.seed(seed)
                        true_beta = np.random.choice(2, size=n_features, p=np.array([0.5, 0.5]))
                        if sum(true_beta) == 0:
                            true_beta[0] = 1
                        np.random.seed(2 + 5 * seed)
                        true_gamma = np.random.choice(2, size=n_features, p=np.array([0.2, 0.8])) * true_beta

                        problem, true_model_parameters = LMEProblem.generate(**problem_parameters,
                                                                             beta=true_beta,
                                                                             gamma=true_gamma,
                                                                             seed=seed)
                        x, y, columns_labels = problem.to_x_y()

                        model_params = default_params.copy()
                        model_params.update(local_params)

                        model = model_constructor(**model_params,
                                                  nnz_tbeta=sum(true_beta),  # only L0-methods make use of those.
                                                  nnz_tgamma=sum(true_gamma))
                        if not (model_name.startswith("L0") or model_name.endswith("SR3")):
                            params = {
                                "lam": loguniform(1e-1, 3e2)
                            }
                            selector = RandomizedSearchCV(estimator=model,
                                                          param_distributions=params,
                                                          n_iter=10,
                                                          cv=LMEStratifiedShuffleSplit(n_splits=2, test_size=0.5,
                                                                                       random_state=seed,
                                                                                       columns_labels=columns_labels),
                                                          scoring=lambda clf, x, y: -clf.get_information_criterion(x, y,
                                                                                                                   columns_labels=columns_labels,
                                                                                                                   ic="muller_ic"),
                                                          random_state=seed,
                                                          n_jobs=20
                                                          )

                            selector.fit(x, y, columns_labels=columns_labels)
                            model = selector.best_estimator_
                        else:
                            model.fit_problem(problem)

                        y_pred = model.predict_problem(problem)
                        explained_variance = explained_variance_score(y, y_pred)
                        mse = mean_squared_error(y, y_pred)

                        coefficients = model.coef_
                        maybe_tbeta = coefficients["beta"]
                        maybe_tgamma = coefficients["gamma"]
                        fixed_effects_accuracy = accuracy_score(true_beta, abs(maybe_tbeta) > 1e-2)
                        random_effects_accuracy = accuracy_score(true_gamma, abs(maybe_tgamma) > 1e-2)

                        self.assertGreaterEqual(explained_variance, min_explained_variance,
                                                msg=f"{model_name}: Explained variance is too small: {explained_variance} < {min_explained_variance} (seed={seed})")
                        self.assertGreaterEqual(max_mse, mse,
                                                msg=f"{model_name}: MSE is too big: {max_mse} > {mse} (seed={seed})")
                        self.assertGreaterEqual(fixed_effects_accuracy, fixed_effects_min_accuracy,
                                                msg=f"{model_name}: Fixed Effects Selection Accuracy is too small: {fixed_effects_accuracy} < {fixed_effects_min_accuracy}  (seed={seed})")
                        self.assertGreaterEqual(random_effects_accuracy, random_effects_min_accuracy,
                                                msg=f"{model_name}: Random Effects Selection Accuracy is too small: {random_effects_accuracy} < {random_effects_min_accuracy} (seed={seed})")

        return None

    def test_score_function(self):
        # this is only a basic test which checks R^2 in two points: nearly perfect prediction and constant prediction.
        models_to_test = {
            "L0": L0LmeModel,
            "L1": L1LmeModel,
            "L0SR3": L0LmeModelSR3,
            "L1SR3": L1LmeModelSR3,
        }
        problem_parameters = {
            "groups_sizes": [20, 5, 10, 50],
            "features_labels": [FIXED_RANDOM] * 3,
            "fit_fixed_intercept": True,
            "fit_random_intercept": True,
            "features_covariance_matrix": np.array([
                [1, 0, 0],
                [0, 1, 0.7],
                [0, 0.7, 1]
            ]),
            "obs_var": 0.1,
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
        }

        problem, true_model_parameters = LMEProblem.generate(**problem_parameters, seed=42)
        x, y, column_labels = problem.to_x_y()
        for model_name, model_constructor in models_to_test.items():
            with self.subTest(model_name=model_name):
                model = model_constructor(**model_parameters)
                model.fit_problem(problem)
                model.coef_["beta"] = true_model_parameters["beta"]
                model.coef_["random_effects"] = random_effects_to_matrix(true_model_parameters["random_effects"])
                good_score = model.score(x, y, columns_labels=column_labels,
                                         fit_fixed_intercept=True, fit_random_intercept=True)
                self.assertGreaterEqual(good_score, 0.9)
                model.coef_["beta"] = np.zeros(4)
                model.coef_["random_effects"] = np.zeros((4, 4))
                bad_score = model.score(x, y, columns_labels=column_labels,
                                        fit_fixed_intercept=True, fit_random_intercept=True)
                self.assertLessEqual(abs(bad_score), 0.1)


if __name__ == '__main__':
    unittest.main()
