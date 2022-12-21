# This code implements linear mixed-effects problem generator and related subroutines.
# Copyright (C) 2020 Aleksei Sholokhov, aksh@uw.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Representations of datasets that are compatible with skmixed's models
"""

import warnings
from typing import Union, Sized, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y

GROUP = "group"
INTERCEPT = "intercept"
VARIANCE = "variance"
FIXED = "fixed"
RANDOM = "random"
FIXED_RANDOM = "fixed+random"


class Problem(object):
    """
    Template class for various representations of datasets.
    """

    def __init__(self, **kwargs):
        """
        Initializes the class
        Parameters
        ----------
        kwargs:
            anything needed
        """
        pass

    def from_x_y(self, x, y, **kwargs):
        """
        Creates Problem from matrices X and Y

        Parameters
        ----------
        x: ndarray, (n, p)
            data matrix
        y: ndarray (n, )
            target variable
        kwargs:
            anything needed

        Returns
        -------
        Problem
        """
        pass

    def to_x_y(self, **kwargs):
        """
        Converts its internal representation into the (X, y) dataset

        Parameters
        ----------
        kwargs:
            anything needed

        Returns
        -------
            Matrices X (n, p) and y (n, )
        """
        pass


default_generator_parameters = {
    "min_elements_per_group": 1,
    "max_elements_per_group": 1000,
    "min_groups": 1,
    "max_groups": 10,
    "min_features": 1,
    "max_features": 10,
    "num_attempts_to_generate_cat_features": 1000,
}


class LMEProblem(Problem):
    """
    Helper class which implements Linear Mixed-Effects models' abstractions over a given dataset.

    It also can generate random problems with specific characteristics.
    """

    def __init__(self,
                 fixed_features: List[np.ndarray],
                 random_features: List[np.ndarray],
                 obs_vars: Union[int, float, np.ndarray],
                 group_labels: np.ndarray,
                 intercept_label: str,
                 column_labels: List[Tuple[int, int]],
                 order_of_objects: np.ndarray,
                 answers=None,
                 fe_columns=None,
                 re_columns=None,
                 fe_regularization_weights=None,
                 re_regularization_weights=None):
        """
        Constructor for LMEProblem class. It is meant to be used by library's internals.
        Check out class methods like from_x_y() and from_dataframe() below that are designed
        to be used by users for creating LMEProblems.
        """
        super(LMEProblem, self).__init__()

        self.fixed_features = fixed_features
        self.answers = answers
        self.random_features = random_features
        self.obs_vars = obs_vars

        self.groups_sizes = [x.shape[0] for x in fixed_features]
        self.num_groups = len(self.groups_sizes)
        self.num_obs = sum(self.groups_sizes)
        self.group_labels = group_labels
        self.intercept_label = intercept_label
        self.column_labels = column_labels
        self.order_of_objects = order_of_objects

        self.num_features = sum([label in (FIXED, RANDOM, FIXED_RANDOM) for label in column_labels]) + int(
            intercept_label is not None)
        self.num_fixed_features = (sum([label in (FIXED, FIXED_RANDOM) for label in column_labels]) +
                                   int((intercept_label == FIXED) or (intercept_label == FIXED_RANDOM)))
        self.num_random_features = (sum([label in (RANDOM, FIXED_RANDOM) for label in column_labels]) +
                                    int((intercept_label == RANDOM) or (intercept_label == FIXED_RANDOM)))

        self.fixed_features_columns = fe_columns
        self.random_features_columns = re_columns
        self.fe_regularization_weights = fe_regularization_weights
        self.re_regularization_weights = re_regularization_weights

    def __iter__(self):
        """
        Iterator initializer

        Returns
        -------
        self
        """
        self.__iteration_pos = 0
        return self

    def __next__(self):
        """
        Iterates over (x, y, z, l) for each group

        Returns
        -------
        Iterator for a tuple of four elements for group i:

            - xi, ndarray (ni, p), fixed effects
            - yi, ndarray (ni, ), target variable
            - zi, ndarray (ni, q), random effects
            - li, ndarray (ni, ), variances of observation errors
        """
        j = self.__iteration_pos
        if j < len(self.fixed_features):
            self.__iteration_pos += 1
            if self.answers is None:
                answers = None
            else:
                answers = self.answers[j]
            return self.fixed_features[j], answers, self.random_features[j], self.obs_vars[j]
        else:
            raise StopIteration

    @staticmethod
    def generate(groups_sizes: Optional[List[Optional[int]]] = None,
                 features_labels: Optional[List[str]] = None,
                 fit_fixed_intercept: bool = False,
                 fit_random_intercept: bool = False,
                 features_covariance_matrix: Optional[np.ndarray] = None,
                 obs_var: Optional[Union[int, float, Sized]] = 0.1,
                 beta: Optional[np.ndarray] = None,
                 gamma: Optional[np.ndarray] = None,
                 true_random_effects: Optional[np.ndarray] = None,
                 as_x_y=False,
                 return_true_model_coefficients: bool = True,
                 seed: int = None,
                 generator_params: dict = None,
                 chance_missing: float = 0.0,
                 chance_outlier: float = 0.0,
                 outlier_multiplier: float = 5.0,
                 distribution="normal",
                 ):
        """

        Generates a random mixed-effects problem with given parameters.

        The model is::

            Y_i = X_i*β + Z_i*u_i + 𝜺_i,

            where

            u_i ~ 𝒩(0, diag(𝛄)),

            𝜺_i ~ 𝒩(0, diag(variance))

        Parameters
        ----------

        groups_sizes : List, Optional
            List of groups sizes. If None then generates it from U[1, 1000]^k where k ~ U[1, 10]

        features_labels : List, Optional
            List of features labels which define whether a role of features in the problem: "fixed" -- fixed only,
            "random" -- random only, "fixed+random" -- both.
            Does NOT include intercept (it's handled with fit_random_intercept parameter).
            If None then generates a random list from U[1, 4]^k where k ~ U[1, 10]

        fit_fixed_intercept : bool, default is False
            If True then the model adds intercept to the set of fixed features. Intercept should not be
             in the features_covariance_matrix or features_labels.

        fit_random_intercept : bool, default is False
            True if the intercept is a random parameter as well. Intercept should never be
             in the features_covariance_matrix or features_labels.

        features_covariance_matrix : np.ndarray, Optional, Symmetric and PSD
            Covariance matrix of the features from features labels (columns from the dataset to be generated).
            If None then defaults to the identity matrix, in which case all features are independent.
            Should be the size of len(features_labels).

        obs_var : float or np.ndarray
            Variances of measurement errors. Can be:

                - float : In this case all errors for all groups have the same variance.
                -   | np.array of length equal to the number of groups : In this case each group has its own variance
                    | of the measurement errors, and it is the same for all objects within a group.
                -   | stds : np.array of length equal to the number of objects in all groups cumulatively.
                    | In this case every object has its own variance.

            Raise ValueError if obs_var has some other length then above.

        beta : np.ndarray
            True vector of fixed effects. Should be equal to the number of fixed features in the features_labels
            plus one (intercept).
            If None then it's generated randomly from U[0, 1]^k where k is the number of fixed features plus intercept.

        gamma : np.ndarray
            True vector of random effects. Should be equal to the number of random features in the features_labels
            plus one if fit_random_intercept is True.
            If None then it's generated randomly from U[0, 1]^k where k is the number of random effects plus (maybe)
            intercept.

        true_random_effects: np.ndarray
            True random effects. Should be of a shape=(m, k) where m is the length of gamma, k is the number of groups.
            If None then generated according to the model: u_i ~ 𝒩(0, diag(𝛄)).

        as_x_y : bool, default is False
            If True, returns the data in the form of tuple of matrices (X, y). Otherwise returns an instance
            of the respective class.

        return_true_model_coefficients : bool, default is True
            If True, the second return argument is a dict with true model coefficients: beta, gamma, random effects and
            true values of measurements errors, otherwise returns None.

        seed : int, default is None
            If given, initializes the global Numpy random generator with this seed.

        generator_params : dict
            Dictionary with the parameters of the problem generator,
            like min-max bounds for the number of groups and objects.
            If None then the default one is used (see at the beginning of this file).

        distribution : str
            which distribution is used for generating features: "normal" or "uniform"

        chance_outlier : float, from 0 to 1
            chance that a selected value in data matrix is an outlier. If so, it gets 
            multiplied by outlier_multiplier

        outlier_multiplier : float
            magnitude of the outliers

        chance_missing : float, from 0 to 1
            chance that a selected value is going to be missing from the dataset, 
            in which case it's set to 0.

        Returns
        -------
        problem : LMEProblem
            Generated problem

        true_parameters : dict, optional
            True parameters for generated problem:

                - "beta" : true beta,
                - "gamma" : true gamma,
                - "per_group_coefficients": true per group coefficients (b such that y = Xb, where X is from to_x_y())
                - "active_categorical_set": set of categorical features which were used for true latent group division
                - "true_group_labels": labels from true latent group division
                - "random_effects": true random effects
                - "errors": true errors
                - "true_rmse": loss value when true beta, gamma and random effects are used.
        """

        if generator_params is None:
            generator_params = default_generator_parameters

        if seed is not None:
            np.random.seed(seed)

        if features_labels is None:
            if features_covariance_matrix is not None:
                len_features_labels = features_covariance_matrix.shape[0]
            else:
                len_features_labels = np.random.randint(generator_params["min_features"],
                                                        generator_params["max_features"])
            features_labels = np.random.choice((FIXED, RANDOM, FIXED_RANDOM), len_features_labels).tolist()

        num_features_to_generate = len(features_labels)
        if num_features_to_generate == 0 and not fit_fixed_intercept and not fit_random_intercept:
            raise ValueError("Problem is empty: no features and no fixed or random intercept.")

        fixed_features_idx = np.array([i for i, label in enumerate(features_labels) if
                                       label in (FIXED, FIXED_RANDOM)])
        random_features_idx = np.array([i for i, label in enumerate(features_labels) if
                                        label in (RANDOM, FIXED_RANDOM)])

        num_fixed_features = len(fixed_features_idx) + (1 if fit_fixed_intercept else 0)
        num_random_features = len(random_features_idx) + (1 if fit_random_intercept else 0)

        if num_fixed_features == 0:
            raise ValueError("Zero fixed features requested for the problem. Should be at least one.")

        if beta is None:
            beta = np.random.rand(num_fixed_features)
        else:
            assert beta.shape[0] == num_fixed_features, \
                (f"beta has the size {beta.shape[0]}, but the number of fixed effects," +
                 f" including intercept, is {num_fixed_features}")
        if gamma is None:
            gamma = np.random.rand(num_random_features)
        else:
            assert gamma.shape[0] == num_random_features, \
                "gamma has the size %d, but the number of random effects, including intercept, is %s" % (
                    gamma.shape[0],
                    num_random_features
                )

        # features covariance matrix describes covariances only presented in features_labels (meaningful features),
        # so we exclude the intercept feature when we generate random correlated data.
        if features_covariance_matrix is None:
            features_covariance_matrix = np.eye(num_features_to_generate)
        else:
            assert features_covariance_matrix.shape[0] == num_features_to_generate == \
                   features_covariance_matrix.shape[1], ("features_covariance_matrix should be n*n " +
                                                         "where n is the total number of features " +
                                                         "(excluding intercept)")

        data = {
            'fixed_features': [],
            'random_features': [],
            'answers': [],
            'obs_vars': [],
        }

        if groups_sizes is None:
            num_groups = np.random.randint(generator_params["min_groups"], generator_params["max_groups"])
            groups_sizes = np.random.randint(generator_params["min_elements_per_group"],
                                             generator_params["max_elements_per_group"],
                                             num_groups)
        else:
            num_groups = len(groups_sizes)
            for i, group_size in enumerate(groups_sizes):
                if group_size is None:
                    groups_sizes[i] = np.random.randint(generator_params["min_elements_per_group"],
                                                        generator_params["max_elements_per_group"])

        num_objects = sum(groups_sizes)

        if num_features_to_generate > 0:
            if distribution == "normal":
                features_data = np.random.multivariate_normal(np.zeros(num_features_to_generate),
                                                              features_covariance_matrix,
                                                              num_objects)
            elif distribution == "uniform":
                features_data = np.random.uniform(-2, 2, (num_objects, num_features_to_generate))
            else:
                warnings.warn(f"Unknown distribution {distribution}, default to 'normal'")
                features_data = np.random.multivariate_normal(np.zeros(num_features_to_generate),
                                                              features_covariance_matrix,
                                                              num_objects)

            all_data_in_one_matrix = features_data
        else:
            all_data_in_one_matrix = np.ones(num_objects).reshape((-1, 1))

        random_effects_list = []
        errors_list = []
        order_of_objects = []
        true_group_labels = np.zeros(num_objects)
        true_rmse = 0
        start = 0

        for i, group_size in enumerate(groups_sizes):
            group_label = i
            group_features = all_data_in_one_matrix[start:start + group_size, :]

            # generate outliers and missing values after the answers are generated
            missing_and_outliers_mask = np.random.choice([0, 1, outlier_multiplier],
                                                         size=group_features.shape,
                                                         p=[
                                                             chance_missing,
                                                             1 - chance_missing - chance_outlier,
                                                             chance_outlier
                                                         ])
            group_features *= missing_and_outliers_mask

            if len(fixed_features_idx) > 0:
                group_fixed_features = group_features[:, fixed_features_idx]
                if fit_fixed_intercept:
                    group_fixed_features = np.hstack([np.ones((group_size, 1)), group_fixed_features])
            else:
                # we already checked above that there must be an intercept in this case
                group_fixed_features = np.ones((group_size, 1))

            if num_random_features > 0:
                if len(random_features_idx) > 0:
                    group_random_features = group_features[:, random_features_idx]
                    if fit_random_intercept:
                        group_random_features = np.hstack([np.ones((group_size, 1)), group_random_features])
                else:
                    group_random_features = np.ones((group_size, 1))
                random_effects = None
                if true_random_effects is not None:
                    random_effects = dict(true_random_effects).get(group_label, None)
                if random_effects is None:
                    random_effects = np.random.multivariate_normal(np.zeros(num_random_features), np.diag(gamma))

            else:
                group_random_features = np.array(0)
                random_effects = np.array(0)

            if isinstance(obs_var, np.ndarray):
                if obs_var.shape[0] == sum(groups_sizes):
                    variance = obs_var[start:group_size]
                elif obs_var.shape[0] == num_groups:
                    variance = obs_var[i]
                else:
                    raise ValueError("len(obs_var) should be either num_groups or sum(groups_sizes)")

            elif isinstance(obs_var, (float, int)):
                variance = obs_var
            else:
                raise ValueError("obs_var is not an array or int/float.")

            group_variances = variance
            group_errors = np.random.normal(loc=0, scale=np.sqrt(variance), size=group_size)
            group_answers = group_fixed_features.dot(beta) + group_random_features.dot(random_effects) + group_errors

            true_group_labels[start: start + group_size] = i

            random_effects_list.append((group_label, random_effects))

            order_of_objects += list(range(start, start + group_size))
            start += group_size

            true_rmse += np.linalg.norm(group_errors) ** 2
            data['fixed_features'].append(group_fixed_features)
            data['random_features'].append(group_random_features)
            data['answers'].append(group_answers)
            data['obs_vars'].append(np.ones(group_size) * group_variances)

            errors_list.append(group_errors)

        data['group_labels'] = np.arange(start=0, stop=num_groups, step=1)
        data['order_of_objects'] = np.array(order_of_objects)

        if fit_fixed_intercept and fit_random_intercept:
            intercept_label = FIXED_RANDOM
        elif fit_fixed_intercept:
            intercept_label = FIXED
        elif fit_random_intercept:
            intercept_label = RANDOM
        else:
            intercept_label = None

        #  [group] + [features] + [variance]
        all_columns_labels = [GROUP] + features_labels + [VARIANCE]
        data['fe_columns'] = (["intercept"] if fit_fixed_intercept else []) + [f"x{i}" for i in fixed_features_idx]
        data['re_columns'] = (["intercept"] if fit_random_intercept else []) + [f"x{i}" for i in random_features_idx]
        data['column_labels'] = all_columns_labels
        data['intercept_label'] = intercept_label
        data['fe_regularization_weights'] = np.ones(num_fixed_features)
        data['re_regularization_weights'] = np.ones(num_random_features)

        # We pivot the groups back to the original group division
        generated_problem = LMEProblem(**data)
        generated_problem = generated_problem.to_x_y() if as_x_y else generated_problem

        if return_true_model_coefficients:
            per_group_coefficients = get_per_group_coefficients(beta,
                                                                random_effects_list,
                                                                labels=np.array(all_columns_labels))
            true_parameters = {
                "beta": beta,
                "gamma": gamma,
                "per_group_coefficients": per_group_coefficients,
                "true_group_labels": true_group_labels,
                "random_effects": random_effects_list,
                "errors": errors_list,
                "true_rmse": true_rmse
            }
            return generated_problem, true_parameters
        else:
            return generated_problem, None

    @staticmethod
    def from_x_y(x: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 columns: List[str] = None,
                 columns_labels: List[str] = None,
                 fit_fixed_intercept: bool = False,
                 fit_random_intercept: bool = False,
                 must_include_fe: List[str] = None,
                 must_include_re: List[str] = None,
                 **kwargs):
        """
        Transforms matrices x (data) and y(answers) into an instance of LMEProblem

        Parameters
        ----------
        x: array-like, shape = [m,n]
            Data.

        y: array-like, shape = [m]
            Answers.
        
        columns: List[str]
            List of columns names

        columns_labels :  List[str]
            List of column labels. There shall be only one column of group labels and answers STDs.

                - "fixed" : fixed effect
                - "random" : random effect
                - "fixed+random" : both fixed and random,
                - "group" : groups labels
                - "variance" : answers standard deviations
                -   | "intercept" : intercept column (fixed or random intercept is controlled by "fit_fixed_intercept"
                    | and "fit_random_intercept" respectively.

        fit_fixed_intercept: bool, default = True
            Whether to add an intercept as a fixed feature

        fit_random_intercept: bool, default = True
            Whether to add an intercept as a random feature.

        must_include_re: List[str]
            List of fixed effects for which any effect of sparsity-promoting regularizers
            should be disabled. NB: it does not guarantee the inclusion of this feature
            to the ultimate model.
            
        must_include_fe: List[str]
            Same for random effects

        kwargs:
            It's not used now, but it's left here for future.

        Returns
        -------
        problem: LMEProblem
            an instance of LMEProblem build on the given data.
        """

        if columns_labels is None:
            columns_labels = [GROUP] + [FIXED] * x.shape[1] + [VARIANCE]
            x = np.hstack([np.array([0] * x.shape[0]).reshape(-1, 1),
                           x,
                           np.array([1] * x.shape[0]).reshape(-1, 1)])

        if y is not None:
            x, y = check_X_y(x, y)
        assert set(columns_labels).issubset(
            (GROUP, VARIANCE, FIXED, RANDOM, FIXED_RANDOM)), \
            f"Only {GROUP}, {VARIANCE}, {FIXED}, {RANDOM}, and {FIXED_RANDOM} are allowed in columns_labels"
        assert len(columns_labels) == x.shape[1], "len(columns_labels) != x.shape[1] (not all columns are labelled)"
        # take the index of a column that stores group labels
        group_labels_idx = [i for i, label in enumerate(columns_labels) if label == GROUP]
        assert len(group_labels_idx) == 1, f"There should be only one '{GROUP}' in columns_labels"
        group_labels_idx = group_labels_idx[0]
        obs_variances_idx = [i for i, label in enumerate(columns_labels) if label == VARIANCE]
        assert len(obs_variances_idx) == 1, f"There should be only one '{VARIANCE}' in columns_labels"
        obs_variances_idx = obs_variances_idx[0]
        assert all(x[:, obs_variances_idx] != 0), ("Errors' variances can't be zero." +
                                                   " Check for zeros in the respective column.")
        fixed_features_idx = [i for i, t in enumerate(columns_labels) if t == FIXED or t == FIXED_RANDOM]
        random_features_idx = [i for i, t in enumerate(columns_labels) if t == RANDOM or t == FIXED_RANDOM]

        groups_labels = unique_labels(x[:, group_labels_idx].tolist())

        if columns:
            assert len(columns) == x.shape[1], "'columns' should contain names for all columns"

        if (must_include_fe is not None or must_include_re is not None) and (columns is None):
            raise ValueError(
                "'columns' must be provided when 'not_regularized_fe' or 'not_regularized_re' are provided")

        num_fixed_features = len(fixed_features_idx) + (1 if fit_fixed_intercept else 0)

        if must_include_fe is not None and len(must_include_fe) > 0:
            fe_regularization_weights = [int(INTERCEPT not in must_include_fe)] + \
                                        [int(columns[i] not in must_include_fe) for i in fixed_features_idx]
        else:
            fe_regularization_weights = [1] * num_fixed_features

        num_random_features = len(random_features_idx) + (1 if fit_random_intercept else 0)

        if must_include_re is not None and len(must_include_re) > 0:
            re_regularization_weights = [int(INTERCEPT not in must_include_fe)] + \
                                        [int(columns[i] not in must_include_re) for i in random_features_idx]
        else:
            re_regularization_weights = [1] * num_random_features

        if fit_fixed_intercept & fit_random_intercept:
            intercept_label = FIXED_RANDOM
        elif fit_random_intercept and not fit_fixed_intercept:
            intercept_label = RANDOM
        elif fit_fixed_intercept and not fit_random_intercept:
            intercept_label = FIXED
        else:
            intercept_label = None

        fe_columns = [columns[i] for i in fixed_features_idx] if columns else None
        re_columns = [columns[i] for i in random_features_idx] if columns else None

        data = {
            'fixed_features': [],
            'random_features': [],
            'answers': None if y is None else [],
            'obs_vars': [],
            'group_labels': groups_labels,
            'column_labels': columns_labels,
            'intercept_label': intercept_label,
            'fe_columns': fe_columns,
            're_columns': re_columns,
            'fe_regularization_weights': fe_regularization_weights,
            're_regularization_weights': re_regularization_weights,
        }

        order_of_objects = []
        for label in groups_labels:
            objects_idx = x[:, group_labels_idx] == label
            order_of_objects += np.where(objects_idx)[0].tolist()
            fixed_features = x[np.ix_(objects_idx, fixed_features_idx)]
            # add an intercept column plus real features
            if fit_fixed_intercept:
                data['fixed_features'].append(
                    np.concatenate((np.ones((fixed_features.shape[0], 1)), fixed_features), axis=1))
            else:
                data['fixed_features'].append(fixed_features)
            # same for random effects
            random_features = x[np.ix_(objects_idx, random_features_idx)]
            if fit_random_intercept:
                data['random_features'].append(np.concatenate((np.ones((random_features.shape[0], 1)), random_features),
                                                              axis=1))
            else:
                data['random_features'].append(random_features)
            if y is not None:
                data['answers'].append(y[objects_idx])
            data['obs_vars'].append(x[objects_idx, obs_variances_idx])

        data["order_of_objects"] = order_of_objects

        return LMEProblem(**data)

    def to_x_y(self) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Transforms the problem to the (X, y) form.

        The first row of X is going to be features labels.

        Returns
        -------
        X : np.ndarray
            Features as a matrix
        y : np.ndarray
            Answer as a vector
        """

        all_group_labels = np.repeat(self.group_labels, self.groups_sizes)
        all_fixed_features = np.concatenate(self.fixed_features, axis=0)
        all_random_features = np.concatenate(self.random_features, axis=0)
        all_variances = np.concatenate(self.obs_vars, axis=0)

        untitled_data = np.zeros((all_fixed_features.shape[0], len(self.column_labels)))

        fixed_intercept = (self.intercept_label == FIXED) or (self.intercept_label == FIXED_RANDOM)
        if fixed_intercept:
            fixed_effects_counter = 1
        else:
            fixed_effects_counter = 0

        random_intercept = (self.intercept_label == RANDOM) or (self.intercept_label == FIXED_RANDOM)
        if random_intercept:
            random_effects_counter = 1
        else:
            random_effects_counter = 0

        for i, label in enumerate(self.column_labels):
            if label == GROUP:
                untitled_data[:, i] = all_group_labels
            elif label == FIXED:
                untitled_data[:, i] = all_fixed_features[:, fixed_effects_counter]
                fixed_effects_counter += 1
            elif label == RANDOM:
                untitled_data[:, i] = all_random_features[:, random_effects_counter]
                random_effects_counter += 1
            elif label == FIXED_RANDOM:
                untitled_data[:, i] = all_fixed_features[:, fixed_effects_counter]
                fixed_effects_counter += 1
                random_effects_counter += 1
            elif label == VARIANCE:
                untitled_data[:, i] = all_variances

        untitled_data = untitled_data[np.array(self.order_of_objects).argsort()]
        if self.answers is not None:
            all_answers = np.concatenate(self.answers)
        else:
            all_answers = None
        all_answers = all_answers[np.array(self.order_of_objects).argsort()]
        return untitled_data, all_answers, self.column_labels

    @staticmethod
    def from_dataframe(data: pd.DataFrame,
                       fixed_effects: List[str],
                       random_effects: List[str],
                       groups: str,
                       variance: str,
                       target: str,
                       not_regularized_fe: List[str],
                       not_regularized_re: List[str]
                       ):
        """
        Creates LMEProblem from Pandas dataframe

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe that contains all relevant data
        fixed_effects : List[str]
            List of column names that should be included as fixed effects
        random_effects : List[str]
            List of column names that should be included as random effects
        groups : str
            Name of the column that contains groups labels
        variance : str
            Name of the column that contains observation variances
        target : str
            Name of the column that contains the target variable
        not_regularized_fe : str
            List of fixed effects which corresponding coefficients in the model are not penalized by
            a sparsity-promoting regularizer. Does NOT guarantee that these features are going to be
            included to the final model but significantly increases the chances of it.
        not_regularized_re : str
            List of random effects which corresponding coefficients in the model are not penalized by
            a sparsity-promoting regularizer. Does NOT guarantee that these features are going to be
            included to the final model but significantly increases the chances of it.

        Returns
        -------
            LMEProblem
        """

        for effect in fixed_effects + random_effects + [groups, variance,
                                                        target] + not_regularized_fe + not_regularized_re:
            if effect not in data.columns:
                raise ValueError(f"{effect} is not a column of the data-frame")

        if not all(effect in fixed_effects for effect in not_regularized_fe):
            raise ValueError("All elements from not_regularized_fe should also be in fixed_effects")
        if not all(effect in random_effects for effect in not_regularized_re):
            raise ValueError("All elements from not_regularized_re should also be in random_effects")

        columns = []
        column_labels = []

        for effect in data.columns:
            if (effect not in fixed_effects) and (effect not in random_effects) or (effect == INTERCEPT):
                continue
            elif (effect in fixed_effects) and (effect in random_effects):
                column_labels.append(FIXED_RANDOM)
            elif effect in random_effects:
                column_labels.append(RANDOM)
            else:
                column_labels.append(FIXED)
            columns.append(effect)

        columns = [groups] + columns + [variance]
        column_labels = [GROUP] + column_labels + [VARIANCE]

        x = data[columns].to_numpy()
        y = data[target].to_numpy()
        return LMEProblem.from_x_y(x=x,
                                   y=y,
                                   columns_labels=column_labels,
                                   columns=columns,
                                   fit_fixed_intercept="intercept" in fixed_effects,
                                   fit_random_intercept="intercept" in random_effects,
                                   must_include_fe=not_regularized_fe,
                                   must_include_re=not_regularized_re)


    def to_dataframe(self):
        x, y, column_labels = self.to_x_y()
        data = pd.DataFrame(x, columns=column_labels)
        data['target'] = y
        return data




class LMEStratifiedShuffleSplit:
    """
    Class that generates shuffle splits of the dataset that are stratified by groups
    """

    def __init__(self, columns_labels: List[str], random_state=42, test_size=0.25, n_splits=3):
        """
        Creates LMEStratifiedShuffleSplit

        Parameters
        ----------
        columns_labels :  List[str]
            List of column labels. There shall be only one column of group labels and answers STDs.

                - "fixed" : fixed effect
                - "random" : random effect
                - "fixed+random" : both fixed and random,
                - "group" : groups labels
                - "variance" : answers standard deviations
                -   | "intercept" : intercept column (fixed or random intercept is controlled by "fit_fixed_intercept"
                    | and "fit_random_intercept" respectively.

        random_state : int
            Random seed for the generator
        test_size : float, between 0 and 1
            fraction of the dataset for the test part of splits
        n_splits : int
            number of splits
        """
        self.columns_labels = columns_labels
        self.seed = random_state
        self.test_size = test_size
        self.n_splits = n_splits

    def split(self, x=None, y=None, groups=None):
        """
        Generates splits

        Parameters
        ----------
        x : ndarray, (n, p)
            data matrix
        y : ndarray (n, )
            target variable

        Returns
        -------
        Iterable over tuples (X_train, y_train, X_test, y_test) that are stratified by the group
        """
        check_X_y(x, y)
        splitter = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.seed)
        group_column = np.where([label == GROUP for label in self.columns_labels])[0]
        return splitter.split(x, y=x[:, group_column])

    def get_n_splits(self, x, y, groups):
        return self.n_splits


def get_per_group_coefficients(beta, random_effects, labels):
    """
    Derives per group coefficients from the vectors of fixed and per-cluster random effects.

    Parameters
    ----------
    beta: ndarray, shape=(n,), n is the number of fixed effects.
        Vector of fixed effects.

    random_effects: ndarray or list, shape=(m, k), m groups, k random effects.
        Array of random effects.

    labels :  List[str]
        List of column labels. There shall be only one column of group labels and answers STDs.

            - "fixed" : fixed effect
            - "random" : random effect
            - "fixed+random" : both fixed and random,
            - "group" : groups labels
            - "variance" : answers standard deviations
            -   | "intercept" : intercept column (fixed or random intercept is controlled by "fit_fixed_intercept"
                | and "fit_random_intercept" respectively.

    Returns
    -------
    per_group_coefficients: ndarray, shape=(m, t)
        Array of cluster coefficients: m clusters times t coefficients.
    """
    is_arrays = False
    if all([type(s) == np.ndarray for s in random_effects]):
        random_effects = [(None, u) for u in random_effects]
        is_arrays = True

    per_group_coefficients_list = []

    for i, (label, u) in enumerate(random_effects):
        per_group_coefficients = np.zeros(len(labels))
        fixed_effects_counter = 0
        random_effects_counter = 0

        for j, feature_label in enumerate(labels):
            if feature_label == FIXED:
                per_group_coefficients[j] = beta[fixed_effects_counter]
                fixed_effects_counter += 1
            elif feature_label == RANDOM:
                per_group_coefficients[j] = u[random_effects_counter]
                random_effects_counter += 1
            elif feature_label == FIXED_RANDOM:
                per_group_coefficients[j] = beta[fixed_effects_counter] + u[random_effects_counter]
                fixed_effects_counter += 1
                random_effects_counter += 1
            else:
                continue

        per_group_coefficients_list.append((label, per_group_coefficients))
    if is_arrays:
        return random_effects_to_matrix(per_group_coefficients_list)
    else:
        return per_group_coefficients_list


def random_effects_to_matrix(random_effects):
    """
    Stacks a list of tuples (group: random effects) into an array

    Parameters
    ----------
    random_effects: List[Tuple[Any, ndarray]]
        List of random effects in the format [(group1: effect1), (group2: effects2), ...]

    Returns
    -------
    ndarray of random effects stacked vertically
    """
    return np.array([u for k, u in random_effects])
