from typing import List

import numpy as np
import pandas as pd


class LinearProblem:
    """
    Helper class which implements Linear models' abstractions over a given dataset.

    It also can generate random problems with specific characteristics.
    """

    def __init__(self,
                 a,
                 b,
                 c=None,
                 obs_std=None,
                 regularization_weights=None):
        """
        Constructs LinearProblem -- a helper class that abstracts the data for the models.

        Parameters
        ----------
        a: ndarray (n, p)
            data matrix
        b: ndarray (n, )
            target variable
        obs_std: ndarray (n, )
            variances of mean-zero Gaussian noise for each observation
        regularization_weights: ndarray (n, )
            observation-specific weights for the regularizer. Inverse-proportional to the objects' importance.
        """
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
        """
        Generates a random dataset with a linear dependence between observations and features

        Parameters
        ----------
        num_objects: int
            number of objects (rows) in the dataset
        num_features: int
            number of features (columns) in the dataset
        obs_std: float | ndarray (num_objects, )
            variances of mean-zero Gaussian noise for each observation (array) OR for all observations (float)
        true_x: ndarray (num_features, )
            true vector of coefficients. If None then generates a random one from U[0, 1]^num_features
        seed: int
            random seed
        Returns
        -------
        problem: LinearProblem
            generated problem
        """
        np.random.seed(seed)
        a = np.random.rand(num_objects, num_features)
        a[:, 0] = 1
        x = true_x if true_x is not None else np.random.rand(num_features)
        b = a.dot(x) + obs_std * np.random.randn(num_objects)
        return LinearProblem(a=a, b=b, regularization_weights=np.ones(num_features))

    @staticmethod
    def from_x_y(x, y, c=None, regularization_weights=None):
        """
        Creates a LinearProblem from provided dataset

        Parameters
        ----------
        x: ndarray (n, p)
            design matrix with objects being rows and columns being features
        y: ndarray (n, )
            vector of observations
        c: ndarray (p, p), optional
            matrix C for SR3 relaxation, see the paper. If None then an identity is used.
        regularization_weights: ndarray (n, )
            observation-specific weights for the regularizer. Inverse-proportional to the objects' importance.

        Returns
        -------
        problem: LinearProblem
            problem with provided data inside
        """
        return LinearProblem(a=x, b=y, c=c, regularization_weights=regularization_weights)

    def to_x_y(self):
        """
        Converts LinearProblem class to array representation
        Returns
        -------
        x: ndarray (n, p)
            design matrix with objects being rows and columns being features
        y: ndarray (n, )
            vector of observations

        """
        return self.a, self.b

    @staticmethod
    def from_dataframe(data: pd.DataFrame,
                       features: List[str],
                       target: str,
                       must_include_features: List[str] = None,
                       obs_std: str = None,
                       c=None,
                       ):
        """
        Creates LinearProblem from a Pandas dataframe

        Parameters
        ----------
        data: pd.DataFrame
            pandas dataframe with dataset
        features: List[str]
            list of column names that should be included as features
        target: str
            name of the column containing the observations
        must_include_features: List[str]
            list of column names that are not going to be affected by regularization.
            In other words, list of features that receive regularization_weight=0. All others receive 1.
        obs_std: float | ndarray (num_objects, )
            variances of mean-zero Gaussian noise for each observation (array) OR for all observations (float)

        c: ndarray (p, p), optional
            matrix C for SR3 relaxation, see the paper. If None then an identity is used.


        Returns
        -------
        problem: LinearProblem
            problem with the dataset inside
        """
        n = len(features)
        regularization_weights = [1] * n if not must_include_features \
            else [int(feature not in must_include_features) for feature in features]
        return LinearProblem(a=data[features].to_numpy(),
                             b=data[target].to_numpy(),
                             c=c,
                             regularization_weights=regularization_weights,
                             obs_std=obs_std)
