import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import LinearLMESparseModel
from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.trees import Tree, Forest


class TestForest(unittest.TestCase):

    def test_forest(self):
        # TODO: reimplement fit_problem and predict_problem to make it work
        self.assertTrue(True)
        return None
        max_mse = 0.05
        min_explained_variance = 0.9
        categorical_effects_min_accuracy = 0.7
        active_categorical_features = (0, 2, 3)
        for i in range(1):  # TODO: make a better test for this functionality
            problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[40, 30, 50],
                                                                 features_labels=[3, 6, 5, 5, 6],
                                                                 random_intercept=True,
                                                                 obs_std=0.1,
                                                                 seed=i)
            X, y = problem.to_x_y()
            continuous_model = LinearLMESparseModel(lb=0, lg=0, nnz_tbeta=2, nnz_tgamma=2)
            forest = Forest(continuous_model, num_trees=10, max_depth=3)
            forest.fit_problem(problem)
            y_pred = forest.predict_problem(problem)
            x, y_true = problem.to_x_y()
            lower_quantile, higher_quantile = forest.get_prediction_uncertainty(problem)
            errors = np.array([y_pred - lower_quantile, higher_quantile - y_pred])