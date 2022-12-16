import unittest

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from pysr3.lme.problems import LMEProblem, FIXED_RANDOM
from pysr3.lme.model_selectors import select_covariates, MODELS_NAMES


class TestSelectCovariates(unittest.TestCase):

    def test_feature_selector(self):

        trials = 1

        problem_parameters = {
            "groups_sizes": [20, 15, 10, 50],
            "features_labels": [FIXED_RANDOM] * 3,
            "fit_fixed_intercept": False,
            "fit_random_intercept": False,
            "features_covariance_matrix": np.array([
                [1, 0, 0],
                [0, 1, 0.7],
                [0, 0.7, 1]
            ]),
            "obs_var": 0.1,
        }

        for i in range(trials):
            with self.subTest(i=i):
                for model_name in MODELS_NAMES:
                    with self.subTest(model_name=model_name):
                        true_beta = true_gamma = np.array([1, 0, 1])
                        problem, _ = LMEProblem.generate(**problem_parameters, seed=i,
                                                         beta=true_beta, gamma=true_gamma)
                        x, y, labels = problem.to_x_y()
                        data = pd.DataFrame(x, columns=["group", "x1", "x2", "x3", "variance"])
                        # TODO: figure it out
                        data["se"] = np.sqrt(data["variance"])
                        data["target"] = y
                        select_covariates(df=data,
                                          covs={
                                              "fixed_effects": ["x1", "x2", "x3"],
                                              "random_effects": ["x1", "x2", "x3"]
                                          },
                                          target="target",
                                          variance="se",
                                          group="group",
                                          model_name=model_name
                                          )
                        with open('sel_covs.yaml', 'r') as f:
                            answers = yaml.safe_load(f)
                        self.assertEqual(tuple(answers['fixed_effects']), ("x1", "x3"))
                        self.assertEqual(tuple(answers['random_effects']), ("x1", "x3"))
                if Path('sel_covs.yaml').exists():
                    Path('sel_covs.yaml').unlink()

        return None
