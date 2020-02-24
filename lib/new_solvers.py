import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ridge_regression

class LinearLMESolver(BaseEstimator, RegressorMixin):
    """
    Solve Linear Mixed Effects problem with projected
    """

    def __init__(self):
