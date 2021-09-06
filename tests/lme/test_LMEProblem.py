import unittest

import numpy as np

from pysr3.lme.problems import LMEProblem


class TestLinearLMEProblem(unittest.TestCase):
    def test_correctness(self):
        problem_parameters = {
            "groups_sizes": [20, 5, 10, 50],
            "features_labels": ["fixed+random"] * 3,
            "features_covariance_matrix": np.array([
                [1, 0, 0],
                [0, 1, 0.7],
                [0, 0.7, 1]
            ]),
            "obs_var": 0.1,
        }
        problem, true_parameters = LMEProblem.generate(**problem_parameters,
                                                       seed=0)
        x1, y1, columns_labels = problem.to_x_y()
        problem2 = LMEProblem.from_x_y(x1, y1, columns_labels=columns_labels)
        for i, (x, y, z, l) in enumerate(problem2):
            self.assertTrue(np.allclose(y, x.dot(true_parameters['beta']) + z.dot(
                true_parameters['random_effects'][i][1]) + true_parameters['errors'][i]))

    def test_creation_and_from_to_x_y(self):
        problem, true_parameters = LMEProblem.generate(groups_sizes=[20, 30, 50],
                                                       features_labels=["fixed+random",
                                                                        "fixed+random",
                                                                        "fixed",
                                                                        "random"],
                                                       fit_fixed_intercept=True,
                                                       obs_var=0.1,
                                                       seed=42)
        x1, y1, columns_labels = problem.to_x_y()
        problem2 = LMEProblem.from_x_y(x1, y1, columns_labels=columns_labels, fit_fixed_intercept=True)
        x2, y2, columns_labels_2 = problem2.to_x_y()
        self.assertTrue(np.allclose(x1, x2))
        self.assertTrue(np.allclose(y1, y2))
        self.assertEqual(columns_labels, columns_labels_2)

        test_problem, true_test_parameters = LMEProblem.generate(groups_sizes=[3, 4, 5],
                                                                 features_labels=["fixed+random",
                                                                                  "fixed+random",
                                                                                  "fixed",
                                                                                  "random"],
                                                                 fit_fixed_intercept=True,
                                                                 beta=true_parameters["beta"],
                                                                 gamma=true_parameters["gamma"],
                                                                 true_random_effects=true_parameters[
                                                                     "random_effects"],
                                                                 obs_var=0.1,
                                                                 seed=43)

        self.assertTrue(np.allclose(true_parameters["beta"], true_test_parameters["beta"]))
        self.assertTrue(np.allclose(true_parameters["gamma"], true_test_parameters["gamma"]))
        test_us = dict(true_test_parameters["random_effects"])
        for k, u1 in true_parameters["random_effects"]:
            u2 = test_us.get(k, None)
            if u2 is not None:
                self.assertTrue(np.allclose(u1, u2))

    def test_creation_from_no_data(self):
        problem, true_parameters = LMEProblem.generate(groups_sizes=[4, 5, 10],
                                                       features_labels=[],
                                                       fit_fixed_intercept=True,
                                                       fit_random_intercept=True,
                                                       obs_var=0.1,
                                                       seed=42)

        self.assertEqual(len(true_parameters["beta"]), 1, "Beta should be of len = 1 for no-data problem")
        self.assertEqual(len(true_parameters["gamma"]), 1), "Gamma should be of len = 1 for no-data problem"
        self.assertTrue(np.all([np.all(x == 1) and np.all(z == 1) for x, y, z, l in
                                problem])), "All fixed and random features should be 1 for no-data problem"

    def test_from_to_xy_preserves_dataset_structure(self):
        study_sizes = [20, 15, 10]
        num_studies = sum(study_sizes)
        num_fixed_features = 6
        num_random_features = 4
        np.random.seed(42)
        x = np.random.rand(num_studies, 1 + (num_fixed_features - 1) + 1 + (num_random_features - 1) + 1)
        y = np.random.rand(num_studies)
        x[:, 0] = np.repeat([0, 1, 2], study_sizes)
        columns_labels = (["group"] + ["fixed"] * (num_fixed_features - 1) + ["fixed+random"]
                          + ["random"] * (num_random_features - 1) + ["variance"])
        np.random.shuffle(x)
        problem = LMEProblem.from_x_y(x, y, columns_labels=columns_labels)
        x2, y2, columns_labels_2 = problem.to_x_y()
        self.assertTrue(np.all(x2 == x), msg="x is not the same after from/to transformation")
        self.assertTrue(np.all(y2 == y), msg="y is not the same after from/to transformation")
        self.assertTrue(np.all(columns_labels_2 == columns_labels))

    def test_from_dataframe(self):
        problem, true_parameters = LMEProblem.generate(groups_sizes=[40, 30, 50],
                                                       features_labels=["fixed+random"] * 2,
                                                       fit_fixed_intercept=True,
                                                       fit_random_intercept=True,
                                                       obs_var=0.1,
                                                       seed=42,
                                                       )
        x, y, columns_labels = problem.to_x_y()
        import pandas as pd
        data = pd.DataFrame(data=np.hstack([x, y.reshape(-1, 1)]),
                            columns=["groups", "x1", "x2", "obs_var", "target"])
        data["intercept"] = 1
        problem2 = LMEProblem.from_dataframe(data,
                                             fixed_effects=["intercept", "x1", "x2"],
                                             random_effects=["intercept", "x1", "x2"],
                                             variance="obs_var",
                                             target="target",
                                             groups="groups",
                                             must_include_fe=[],
                                             must_include_re=["x1"])
        x2, y2, columns_labels_2 = problem2.to_x_y()
        self.assertTrue(np.all(x == x2))
        self.assertTrue(np.all(y == y2))
        self.assertTrue(np.all(columns_labels == columns_labels_2))


if __name__ == '__main__':
    unittest.main()
