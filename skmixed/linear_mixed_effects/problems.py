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


from typing import Union, Sized, List, Optional, Tuple

import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y

from skmixed.helpers import get_per_group_coefficients


class LMEProblem(object):
    def __init__(self, **kwargs):
        pass

    def from_x_y(self, **kwargs):
        pass

    def to_x_y(self, **kwargs):
        pass


default_generator_parameters = {
    "min_elements_per_group": 1,
    "max_elements_per_group": 1000,
    "min_groups": 1,
    "max_groups": 10,
    "min_features": 1,
    "max_features": 10,
}


class LinearLMEProblem(LMEProblem):

    def __init__(self,
                 fixed_features: List[np.ndarray],
                 random_features: List[np.ndarray],
                 obs_stds: Union[int, float, np.ndarray],
                 group_labels: np.ndarray,
                 column_labels: List[Tuple[int, int]],
                 order_of_objects: np.ndarray,
                 answers=None):
        super(LinearLMEProblem, self).__init__()

        self.fixed_features = fixed_features
        self.answers = answers
        self.random_features = random_features
        self.obs_stds = obs_stds

        self.study_sizes = [x.shape[0] for x in fixed_features]
        self.num_studies = len(self.study_sizes)
        self.num_obs = sum(self.study_sizes)
        self.group_labels = group_labels
        self.column_labels = column_labels
        self.order_of_objects = order_of_objects

        self.num_random_effects = sum([label in (2, 3) for label in column_labels])
        self.num_fixed_effects = sum([label in (1, 3) for label in column_labels])

    def __iter__(self):
        self.__iteration_pos = 0
        return self

    def __next__(self):
        j = self.__iteration_pos
        if j < len(self.fixed_features):
            self.__iteration_pos += 1
            if self.answers is None:
                answers = None
            else:
                answers = self.answers[j]
            return self.fixed_features[j], answers, self.random_features[j], self.obs_stds[j]
        else:
            raise StopIteration

    @staticmethod
    def generate(groups_sizes: Optional[List[Optional[int]]] = None,
                 features_labels: Optional[List[int]] = None,
                 random_intercept: bool = False,
                 features_covariance_matrix: Optional[np.ndarray] = None,
                 # True parameters:
                 obs_std: Optional[Union[int, float, Sized]] = 0.1,
                 beta: Optional[np.ndarray] = None,
                 gamma: Optional[np.ndarray] = None,
                 true_random_effects: Optional[np.ndarray] = None,
                 as_x_y=False,
                 return_true_model_coefficients: bool = True,
                 seed: int = None,
                 generator_params: dict = None,
                 ):
        """
        Generates a random mixed-effects problem with given parameters

        Y_i = X_i*β + Z_i*u_i + 𝜺_i,

        where

        u_i ~ 𝒩(0, diag(𝛄)),

        𝜺_i ~ 𝒩(0, diag(obs_std)

        Parameters
        ----------
        groups_sizes : List, Optional
            List of groups sizes. If None then generates it from U[1, 1000]^k where k ~ U[1, 10]
        features_labels : List, Optional
            List of features labels which define whether a feature is fixed, random, or both:
                1 -- fixed only,
                2 -- random only,
                3 -- both.
            Does NOT include intercept (it's handled with the random_intercept parameter).
            If None then generates a random list from U[1, 4]^k where k ~ U[1, 10]
        random_intercept : bool, default is False
            True if the intercept is a random parameter as well. Intercept is never a part
             of the features_covariance_matrix or features_labels.
        features_covariance_matrix : np.ndarray, Optional, Symmetric and PSD
            Covariance matrix of the features from features labels (columns from the dataset to be generated).
            If None then defaults to the identity matrix, in which case all features are independent.
        obs_std : float or np.ndarray
            Standard deviations of measurement errors. Can be:
                -- float -- in this case all errors for all groups have the same standard deviation.
                -- np.array of length equal to the number of groups. In this case each group has its own standard
                 deviation of the measurement errors, and it is the same for all objects within a group.
                -- np.array of length equal to the number of objects in all groups cumulatively. In this case
                 every object has its own standard deviation.
            Raise ValueError if obs_std has some other length then above.
        beta : np.ndarray
            True vector of fixed effects. Should be equal to the number of fixed features in the features_labels
            plus one (intercept).
            If None then it's generated randomly from U[0, 1]^k where k is the number of fixed features plus intercept.
        gamma : np.ndarray
            True vector of random effects. Should be equal to the number of random features in the features_labels
            plus one if random_intercept is True.
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
            Dictionary with the parameters of the problem generator. If None then the default one is used (see at the
            beginning of this file).

        Returns
        -------
        problem:
                Generated problem
        """

        if generator_params is None:
            generator_params = default_generator_parameters

        if seed is not None:
            np.random.seed(seed)

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

        if features_labels is None:
            if features_covariance_matrix is not None:
                len_features_labels = features_covariance_matrix.shape[0]
            else:
                len_features_labels = np.random.randint(generator_params["min_features"],
                                                        generator_params["max_features"])
            features_labels = np.random.randint(1, 4, len_features_labels).tolist()

        # We add the intercept manually since it is not mentioned in features_labels.
        fixed_effects_idx = np.array([0] + [i + 1 for i, label in enumerate(features_labels) if label in (1, 3)])
        random_effects_idx = np.array(([0] if random_intercept else [])
                                      + [i + 1
                                         for i, label in enumerate(features_labels) if label in (2, 3)])

        num_fixed_effects = len(fixed_effects_idx)
        num_random_effects = len(random_effects_idx)

        if beta is None:
            beta = np.random.rand(num_fixed_effects)
        else:
            assert beta.shape[0] == num_fixed_effects, \
                "beta has the size %d, but the number of fixed effects, including intercept, is %s" % (beta.shape[0],
                                                                                                       num_fixed_effects
                                                                                                       )
        if gamma is None:
            gamma = np.random.rand(num_random_effects)
        else:
            assert gamma.shape[0] == num_random_effects, \
                "gamma has the size %d, but the number of random effects, including intercept, is %s" % (
                    gamma.shape[0],
                    num_random_effects
                )

        data = {
            'fixed_features': [],
            'random_features': [],
            'answers': [],
            'obs_stds': [],
        }

        # features covariance matrix describes covariances only presented in features_labels (meaningful features),
        # so we exclude the intercept feature when we generate random correlated data.
        if features_covariance_matrix is None:
            features_covariance_matrix = np.eye(len(features_labels))
        else:
            assert features_covariance_matrix.shape[0] == len(features_labels) == features_covariance_matrix.shape[1], \
                "features_covariance_matrix should be n*n where n is length of features_labels"

        random_effects_list = []
        errors_list = []
        order_of_objects = []
        start = 0
        for i, size in enumerate(groups_sizes):
            if len(features_labels) > 0:
                all_features = np.random.multivariate_normal(np.zeros(len(features_labels)),
                                                             features_covariance_matrix,
                                                             size)
                # The first feature is always the intercept. It's a 'fake' feature in a sense that it does not appear
                # in the dataset when you export it in the form of (X, y).
                all_features = np.concatenate((np.ones((size, 1)), all_features), axis=1)
            else:
                # if we were not provided with any features then the only column in X is the intercept.
                all_features = np.ones((size, 1))

            fixed_features = all_features[:, fixed_effects_idx]
            random_features = all_features[:, random_effects_idx]
            order_of_objects += list(range(start, start + size))
            start += size

            if true_random_effects is not None:
                random_effects = true_random_effects[i]
            else:
                random_effects = np.random.multivariate_normal(np.zeros(num_random_effects), np.diag(gamma))

            if isinstance(obs_std, np.ndarray):
                if obs_std.shape[0] == sum(groups_sizes):
                    std = obs_std[start:size]
                elif obs_std.shape[0] == num_groups:
                    std = obs_std[i]
                else:
                    raise ValueError("len(obs_std) should be either num_groups or sum(groups_sizes)")
            elif isinstance(obs_std, (float, int)):
                std = obs_std
            else:
                raise ValueError("obs_std is not an array or int/float.")

            errors = np.random.randn(size) * std
            answers = fixed_features.dot(beta) + random_features.dot(random_effects) + errors

            data['fixed_features'].append(fixed_features)
            data['random_features'].append(random_features)
            data['answers'].append(answers)
            data['obs_stds'].append(np.ones(size) * std)
            random_effects_list.append(random_effects)
            errors_list.append(errors)

        data['group_labels'] = np.arange(start=0, stop=num_groups, step=1)
        data['order_of_objects'] = np.array(order_of_objects)
        all_columns_labels = [3 if random_intercept else 1] + features_labels + [4, 0]
        data['column_labels'] = all_columns_labels
        generated_problem = LinearLMEProblem(**data)
        generated_problem = generated_problem.to_x_y() if as_x_y else generated_problem

        if return_true_model_coefficients:
            random_effects = np.array(random_effects_list)

            per_group_coefficients = get_per_group_coefficients(beta,
                                                                random_effects,
                                                                labels=np.array(all_columns_labels))
            true_parameters = {
                "beta": beta,
                "gamma": gamma,
                "per_group_coefficients": per_group_coefficients,
                "random_effects": random_effects,
                "errors": np.array(errors_list)
            }
            return generated_problem, true_parameters
        else:
            return generated_problem, None

    @staticmethod
    def from_x_y(x: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 columns_labels: List[int] = None,
                 random_intercept: bool = True,
                 **kwargs):
        """
        Transforms matrices x (data) and y(answers) into an instance of LinearLMEProblem

        Parameters
        ----------
        x: array-like, shape = [m,n]
            Data.
        y: array-like, shape = [m]
            Answers.
        columns_labels: List, shape = [n], Optional
            A list of column labels which can be 0 (group labels), 1 (fixed effect), 2 (random effect),
             3 (both fixed and random), or 4 (observation standard deviance). There should be only one 0 in the list.
             If it's None then it's assumed that it is the first row of x.
        random_intercept: bool, default = True
            Whether to treat the intercept as a random feature.
        kwargs:
            It's not used now, but it's left here for future.

        Returns
        -------
        problem: LinearLMEProblem
            an instance of LinearLMEProblem build on the given data.
        """

        if columns_labels is None:
            # if no labels were provided we assume the first row of X is column labels
            columns_labels = list(x[0, :].astype(int))
            x = x[1:, :]  # drop the first row to remove the column labels

        if y is not None:
            x, y = check_X_y(x, y)
        assert set(columns_labels).issubset((0, 1, 2, 3, 4)), "Only 0, 1, 2, 3, and 4 are allowed in columns_labels"
        assert len(columns_labels) == x.shape[1], "len(columns_labels) != x.shape[1] (not all columns are labelled)"
        # take the index of a column that stores group labels
        group_labels_idx = [i for i, label in enumerate(columns_labels) if label == 0]
        assert len(group_labels_idx) == 1, "There should be only one 0 in columns_labels"
        group_labels_idx = group_labels_idx[0]
        obs_std_idx = [i for i, label in enumerate(columns_labels) if label == 4]
        assert len(obs_std_idx) == 1, "There should be only one 4 in columns_labels"
        obs_std_idx = obs_std_idx[0]
        assert all(x[:, obs_std_idx] != 0), "Errors' STDs can't be zero. Check for zeros in the respective column."
        features_idx = [i for i, t in enumerate(columns_labels) if t == 1 or t == 3]
        random_features_idx = [i for i, t in enumerate(columns_labels) if t == 2 or t == 3]
        groups_labels = unique_labels(x[:, group_labels_idx])

        data = {
            'fixed_features': [],
            'random_features': [],
            'answers': None if y is None else [],
            'obs_stds': [],
            'group_labels': groups_labels,
            'column_labels': np.array([3 if random_intercept else 1] + columns_labels)
        }

        order_of_objects = []
        for label in groups_labels:
            objects_idx = x[:, group_labels_idx] == label
            order_of_objects += np.where(objects_idx)[0].tolist()
            features = x[np.ix_(objects_idx, features_idx)]
            # add an intercept column plus real features
            # noinspection PyTypeChecker
            data['fixed_features'].append(np.concatenate((np.ones((features.shape[0], 1)), features), axis=1))
            # same for random effects
            random_features = x[np.ix_(objects_idx, random_features_idx)]
            if random_intercept:
                # noinspection PyTypeChecker
                data['random_features'].append(np.concatenate((np.ones((random_features.shape[0], 1)), random_features),
                                                              axis=1))
            else:
                data['random_features'].append(random_features)
            if y is not None:
                data['answers'].append(y[objects_idx])
            data['obs_stds'].append(x[objects_idx, obs_std_idx])

        data["order_of_objects"] = order_of_objects

        return LinearLMEProblem(**data), None

    def to_x_y(self) -> Tuple[np.ndarray, np.ndarray]:
        all_group_labels = np.repeat(self.group_labels, self.study_sizes)
        all_features = np.concatenate(self.fixed_features, axis=0)
        all_random_features = np.concatenate(self.random_features, axis=0)
        all_stds = np.concatenate(self.obs_stds, axis=0)
        untitled_data = np.zeros((all_features.shape[0], len(self.column_labels) - 1))

        fixed_effects_counter = 1
        random_intercept = self.column_labels[0] == 3
        if random_intercept:
            random_effects_counter = 1
        else:
            random_effects_counter = 0

        for i, label in enumerate(self.column_labels[1:]):
            if label == 0:
                untitled_data[:, i] = all_group_labels
            elif label == 1:
                untitled_data[:, i] = all_features[:, fixed_effects_counter]
                fixed_effects_counter += 1
            elif label == 2:
                untitled_data[:, i] = all_random_features[:, random_effects_counter]
                random_effects_counter += 1
            elif label == 3:
                untitled_data[:, i] = all_features[:, fixed_effects_counter]
                fixed_effects_counter += 1
                random_effects_counter += 1
            elif label == 4:
                untitled_data[:, i] = all_stds

        untitled_data = untitled_data[self.order_of_objects, :]
        column_labels = np.array(self.column_labels[1:]).reshape((1, len(self.column_labels[1:])))
        data_with_column_labels = np.concatenate((column_labels, untitled_data), axis=0)
        if self.answers is not None:
            all_answers = np.concatenate(self.answers)
        else:
            all_answers = None
        return data_with_column_labels, all_answers
