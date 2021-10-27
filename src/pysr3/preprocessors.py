import numpy as np

from .linear.problems import LinearProblem


class Preprocessor:

    @staticmethod
    def normalize(problem: LinearProblem, means=None, stds=None):
        if means is None:
            means = np.mean(problem.a, axis=0)
        if stds is None:
            stds = np.std(problem.a, axis=0)
            stds[stds == 0] = 1
        scaled_a = (problem.a - means) / stds
        return LinearProblem(a=scaled_a,
                             b=problem.b,
                             c=problem.c,
                             obs_std=problem.obs_std,
                             regularization_weights=problem.regularization_weights), {"means": means, "stds": stds}

    @staticmethod
    def add_intercept(problem: LinearProblem):
        a_with_intercept = np.c_[np.ones((problem.num_objects, 1)), problem.a]
        if problem.regularization_weights is None:
            regularization_weights = np.ones(problem.num_features)
        else:
            regularization_weights = problem.regularization_weights
        regularization_weights = np.hstack([np.zeros(1), regularization_weights])
        adjusted_c = np.eye(problem.c.shape[0] + 1)
        adjusted_c[1:, 1:] = problem.c
        return LinearProblem(a=a_with_intercept,
                             b=problem.b,
                             c=adjusted_c,
                             obs_std=problem.obs_std,
                             regularization_weights=regularization_weights)