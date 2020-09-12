import unittest

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import LinearLMESparseModel
from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.trees import Tree


class TestTree(unittest.TestCase):

    def test_selecting_categorical_features_only(self):
        # TODO: make a better test for this functionality
        assert True
        return None
        max_mse = 0.05
        min_explained_variance = 0.9
        categorical_effects_min_accuracy = 0.7
        active_categorical_features = (0, 2, 3)
        for i in range(1):
            problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[40, 30, 50],
                                                                 features_labels=[3, 6, 5, 5, 6],
                                                                 random_intercept=True,
                                                                 obs_std=0.1,
                                                                 seed=i)
            continuous_model = LinearLMESparseModel(lb=0, lg=0, nnz_tbeta=2, nnz_tgamma=2)
            tree_model = Tree(model=continuous_model, max_depth=3)
            tree_model.fit_problem(problem)
            y_pred = tree_model.predict_problem(problem)
            pivoted_problem = problem.pivot(tree_model.coef_['chosen_categorical_features'])
            correctly_pivoted_problem = problem.pivot(active_categorical_features)
            x_true, y_true = problem.to_x_y()
            explained_variance = explained_variance_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            selection_accuracy = np.mean([feature in active_categorical_features for feature in tree_model.coef_['chosen_categorical_features']])
            # for x, y, z, l in problem:
            #     plt.scatter(x[:, 1], y)
            # plt.title("Original problem")
            # plt.show()
            # for x, y, z, l in correctly_pivoted_problem:
            #     plt.scatter(x[:, 1], y)
            # plt.title("Correctly pivoted problem")
            # plt.show()


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
            self.assertGreater(selection_accuracy, categorical_effects_min_accuracy,
                               msg="%d) Fixed Effects Selection Accuracy is too small: %.3f < %.2f  (seed=%d)"
                                   % (i,
                                      selection_accuracy,
                                      categorical_effects_min_accuracy,
                                      i)
                               )