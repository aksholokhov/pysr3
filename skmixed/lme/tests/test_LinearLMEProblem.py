import unittest

import numpy as np

from skmixed.lme.problems import LinearLMEProblem


class TestLinearLMEProblem(unittest.TestCase):
    def test_creation_and_from_to_x_y(self):
        problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[20, 30, 50],
                                                             features_labels=[3, 3, 1, 2, 5, 6],
                                                             random_intercept=True,
                                                             obs_std=0.1,
                                                             seed=42)
        x1, y1 = problem.to_x_y()
        problem2, _ = LinearLMEProblem.from_x_y(x1, y1)
        x2, y2 = problem2.to_x_y()
        self.assertTrue(np.all(x1 == x2) and np.all(y1 == y2))
        test_problem, true_test_parameters = LinearLMEProblem.generate(groups_sizes=[3, 4, 5],
                                                                       features_labels=[3, 3, 1, 2, 5, 6],
                                                                       random_intercept=True,
                                                                       beta=true_parameters["beta"],
                                                                       gamma=true_parameters["gamma"],
                                                                       true_random_effects=true_parameters[
                                                                           "random_effects"],
                                                                       obs_std=0.1,
                                                                       seed=43)

        self.assertTrue(np.all(true_parameters["beta"] == true_test_parameters["beta"])
                        and np.all(true_parameters["gamma"] == true_test_parameters["gamma"]))
        for us_subgroups, test_us_subgroups in zip(true_parameters["random_effects"], true_test_parameters["random_effects"]):
            for k, u1 in us_subgroups.items():
                u2 = test_us_subgroups.get(k, None)
                if u2 is not None:
                    self.assertTrue(np.all(u1 == u2))

    def test_creation_from_no_data(self):
        problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[4, 5, 10],
                                                             features_labels=[],
                                                             random_intercept=True,
                                                             obs_std=0.1,
                                                             seed=42)

        self.assertEqual(len(true_parameters["beta"]), 1, "Beta should be of len = 1 for no-data problem")
        self.assertEqual(len(true_parameters["gamma"]), 1), "Gamma should be of len = 1 for no-data problem"
        self.assertTrue(np.all([np.all(x == 1) and np.all(z == 1) for x, y, z, l in
                                problem])), "All fixed and random features should be 1 for no-data problem"

    def test_from_to_xy_preserves_dataset_structure(self):
        study_sizes = [20, 15, 10]
        num_studies = sum(study_sizes)
        num_features = 6
        num_random_effects = 4
        num_categorical_effects = 2
        np.random.seed(42)
        x = np.random.rand(num_studies + 1, 1 + (num_features - 1) + 1 + (num_random_effects - 1) + 1)
        categorical_features = np.random.randint(0, 2, (num_studies+1, num_categorical_effects))
        x = np.concatenate((x, categorical_features), axis=1)
        y = np.random.rand(num_studies)
        x[1:, 0] = np.repeat([0, 1, 2], study_sizes)
        x[0, :] = [0] + [1] * (num_features - 1) + [3] + [2] * (num_random_effects - 1) + [4] + [5]*num_categorical_effects
        np.random.shuffle(x[1:, :])
        problem, true_parameters = LinearLMEProblem.from_x_y(x, y)
        x2, y2 = problem.to_x_y()
        self.assertTrue(np.all(x2 == x), msg="x is not the same after from/to transformation")
        self.assertTrue(np.all(y2 == y), msg="y is not the same after from/to transformation")


if __name__ == '__main__':
    unittest.main()
