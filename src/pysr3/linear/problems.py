from typing import List

import numpy as np
import pandas as pd


class LinearProblem:

    def __init__(self,
                 a,
                 b,
                 c=None,
                 obs_std=None,
                 regularization_weights=None):
        self.a = np.array(a, dtype='float64')
        self.b = np.array(b, dtype='float64')
        self.num_objects = a.shape[0]
        self.num_features = a.shape[1]
        self.c = c if c else np.eye(self.num_features)
        self.obs_std = obs_std
        self.regularization_weights = regularization_weights

    @staticmethod
    def generate(num_objects=100,
                 num_features=10,
                 obs_std=0.1,
                 true_x=None,
                 seed=42):
        np.random.seed(seed)
        a = np.random.rand(num_objects, num_features)
        a[:, 0] = 1
        x = true_x if true_x is not None else np.random.rand(num_features)
        b = a.dot(x) + obs_std * np.random.randn(num_objects)
        return LinearProblem(a=a, b=b, regularization_weights=np.ones(num_features))

    @staticmethod
    def from_x_y(x, y, c=None, regularization_weights=None):
        return LinearProblem(a=x, b=y, c=c, regularization_weights=regularization_weights)

    def to_x_y(self):
        return self.a, self.b

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
        return LinearProblem(a=data[features].to_numpy(),
                             b=data[target].to_numpy(),
                             c=c,
                             regularization_weights=regularization_weights,
                             obs_std=obs_std)
