from typing import List

import numpy as np
import pandas as pd

from pysr3.linear.problems import LinearProblem


class PoissonProblem(LinearProblem):

    @staticmethod
    def generate(num_objects=100,
                 num_features=10,
                 obs_std=0.1,
                 true_x=None,
                 seed=42):
        np.random.seed(seed)
        a = 3*np.random.randn(num_objects, num_features) / num_features
        a[:, 0] = 1
        x = true_x if true_x is not None else np.random.rand(num_features)
        noise = obs_std*np.random.randn(num_objects)
        b = np.array([np.random.poisson(np.exp(ai.dot(x) + eps*0.1)) for ai, eps in zip(a, noise)])
        obs_std = np.ones(num_objects)*obs_std
        return PoissonProblem(a=a, b=b, regularization_weights=np.ones(num_features), obs_std=obs_std)

    @staticmethod
    def from_x_y(x, y, c=None, regularization_weights=None):
        return PoissonProblem(a=x, b=y, c=c, regularization_weights=regularization_weights)

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
