from typing import List

import numpy as np
import pandas as pd


class LinearProblem:

    def __init__(self, a, b, c=None, obs_std=None, must_include_features=None):
        self.a = a
        self.b = b
        self.c = c if c else np.eye(a.shape[1])
        self.obs_std = obs_std
        self.must_include_features = must_include_features

    @staticmethod
    def from_x_y(x, y, c=None):
        return LinearProblem(a=x, b=y, c=c)

    def to_x_y(self):
        return self.a, self.b

    @staticmethod
    def from_dataframe(data: pd.DataFrame,
                       features: List[str],
                       target: str,
                       must_include_features: List[str],
                       obs_std: str = None,
                       c=None
                       ):
        return LinearProblem(a=data[features].to_numpy(),
                             b=data[target].to_numpy(),
                             c=c,
                             must_include_features=must_include_features,
                             obs_std=obs_std)
