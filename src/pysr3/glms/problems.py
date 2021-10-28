from typing import List

import numpy as np
import pandas as pd

from pysr3.linear.problems import LinearProblem


class PoissonProblem(LinearProblem):

    @staticmethod
    def generate(num_objects=100,
                 num_features=10,
                 add_intercept=False,
                 obs_std=0.1,
                 true_x=None,
                 seed=42):
        np.random.seed(seed)
        n = num_features + int(add_intercept)
        a = np.random.rand(num_objects, n)
        if add_intercept:
            a[:, 0] = 1
        if true_x is not None:
            assert len(true_x) == n, "true_x should have length of num_features + ?intercept"
        x = true_x if true_x is not None else np.random.rand(num_features)
        noise = obs_std*np.random.randn(num_objects)
        b = np.array([np.random.poisson(np.exp(ai.dot(x) + eps)) for ai, eps in zip(a, noise)])
        obs_std = obs_std*np.ones(num_objects)
        return PoissonProblem(a=a, b=b, obs_std=obs_std, regularization_weights=np.ones(num_features))

    @staticmethod
    def from_x_y(x, y, c=None, obs_std=None, regularization_weights=None):
        return PoissonProblem(a=x, b=y, c=c, obs_std=obs_std, regularization_weights=regularization_weights)

    @staticmethod
    def from_dataframe(data: pd.DataFrame,
                       features: List[str],
                       target: str,
                       must_include_features: List[str] = None,
                       obs_std: str = None,
                       c=None,
                       ):
        n = len(features)
        regularization_weights = [1] * n if not must_include_features \
            else [int(feature not in must_include_features) for feature in features]
        return PoissonProblem(a=data[features].to_numpy(),
                              b=data[target].to_numpy(),
                              c=c,
                              regularization_weights=regularization_weights,
                              obs_std=obs_std)
