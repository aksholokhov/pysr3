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
    "num_attempts_to_generate_cat_features": 1000,
}


class LinearLMEProblem(LMEProblem):
    """
    Helper class which implements Linear Mixed-Effects models' abstractions over a given dataset.

    It also can generate random problems with specific characteristics.
    """

    def __init__(self,
                 fixed_features: List[np.ndarray],
                 random_features: List[np.ndarray],
                 obs_stds: Union[int, float, np.ndarray],
                 group_labels: np.ndarray,
                 column_labels: List[Tuple[int, int]],
                 order_of_objects: np.ndarray,
                 categorical_features: List[np.ndarray] = None,
                 answers=None,
                 categorical_features_bootstrap_idx=None):

        super(LinearLMEProblem, self).__init__()

        self.fixed_features = fixed_features
        self.answers = answers
        self.categorical_features = categorical_features
        self.random_features = random_features
        self.obs_stds = obs_stds

        self.groups_sizes = [x.shape[0] for x in fixed_features]
        self.num_groups = len(self.groups_sizes)
        self.num_obs = sum(self.groups_sizes)
        self.group_labels = group_labels
        self.column_labels = column_labels
        self.order_of_objects = order_of_objects

        self.num_random_effects = sum([label in (2, 3) for label in column_labels])
        self.num_fixed_effects = sum([label in (1, 3) for label in column_labels])
        self.num_categorical_features = sum([label == 5 for label in column_labels])

        self.categorical_features_bootstrap_idx = categorical_features_bootstrap_idx

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

        Generates a random mixed-effects problem with given parameters.

        The model is::

            Y_i = X_i*β + Z_i*u_i + 𝜺_i,

            where

            u_i ~ 𝒩(0, diag(𝛄)),

            𝜺_i ~ 𝒩(0, diag(obs_std)

        Parameters
        ----------
        groups_sizes : List, Optional
            List of groups sizes. If None then generates it from U[1, 1000]^k where k ~ U[1, 10]

        features_labels : List, Optional
            List of features labels which define whether a role of features in the problem: 1 -- fixed only,
            2 -- random only, 3 -- both, 5 -- categorical (active), 6 -- categorical (inactive).
            Does NOT include intercept (it's handled with the random_intercept parameter).
            If None then generates a random list from U[1, 4]^k where k ~ U[1, 10]

        random_intercept : bool, default is False
            True if the intercept is a random parameter as well. Intercept is never a part
            of the features_covariance_matrix or features_labels.

        features_covariance_matrix : np.ndarray, Optional, Symmetric and PSD
            Covariance matrix of the features from features labels (columns from the dataset to be generated).
            If None then defaults to the identity matrix, in which case all features are independent.
            Should be the size of len(features_labels).

        obs_std : float or np.ndarray
            Standard deviations of measurement errors. Can be:

                - float : In this case all errors for all groups have the same standard deviation std.
                -   | np.array of length equal to the number of groups : In this case each group has its own standard
                    | deviation of the measurement errors, and it is the same for all objects within a group.
                -   | stds : np.array of length equal to the number of objects in all groups cumulatively.
                    | In this case every object has its own standard deviation.

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
            Dictionary with the parameters of the problem generator, like min-max bounds for the number of groups and objects.
            If None then the default one is used (see at the beginning of this file).

        Returns
        -------
        problem : LinearLMEProblem
            Generated problem
        true_parameters : dict, optional
            True parameters for genrated problem:
                - "beta" : true beta,
                - "gamma" : true gamma,
                - "per_group_coefficients": true per group coefficients (b such that y = Xb, where X is from to_x_y())
                - "active_categorical_set": set of categorical features which were used for true latent group division
                - "true_group_labels": labels from true latent group division
                - "random_effects": true random effects
                - "errors": true errors
                - "reference_loss_value": loss value when true beta, gamma and random effects are used.
        """

        if generator_params is None:
            generator_params = default_generator_parameters

        if seed is not None:
            np.random.seed(seed)

        if features_labels is None:
            # This won't generate categorical features, you need to provide them explicitly
            if features_covariance_matrix is not None:
                len_features_labels = features_covariance_matrix.shape[0]
            else:
                len_features_labels = np.random.randint(generator_params["min_features"],
                                                        generator_params["max_features"])
            features_labels = np.random.randint(1, 4, len_features_labels).tolist()

        categorical_features_idx = np.array([i + 1 for i, label in enumerate(features_labels) if label in (5, 6)])
        active_categorical_features_idx = np.array([i + 1 for i, label in enumerate(features_labels) if label in (5,)])
        num_categorical_features = len(categorical_features_idx)

        continuous_features_idx = np.array(
            [0] + [i + 1 for i, label in enumerate(features_labels) if label in (1, 2, 3)])
        num_continuous_features = len(continuous_features_idx)
        # We calculate continuous features idxes like other feature don't exist
        # because we need these structures for slicing over continuous features
        continuous_features_labels = [l for l in features_labels if l in (1, 2, 3)]
        # We add the intercept manually since it is not mentioned in features_labels.
        fixed_features_idx = np.array(
            [0] + [i + 1 for i, label in enumerate(continuous_features_labels) if label in (1, 3)])
        random_features_idx = np.array(([0] if random_intercept else [])
                                       + [i + 1
                                          for i, label in enumerate(continuous_features_labels) if label in (2, 3)])
        num_fixed_features = len(fixed_features_idx)
        num_random_features = len(random_features_idx)

        if beta is None:
            beta = np.random.rand(num_fixed_features)
        else:
            assert beta.shape[0] == num_fixed_features, \
                "beta has the size %d, but the number of fixed effects, including intercept, is %s" % (beta.shape[0],
                                                                                                       num_fixed_features
                                                                                                       )
        if gamma is None:
            gamma = np.random.rand(num_random_features)
        else:
            assert gamma.shape[0] == num_random_features, \
                "gamma has the size %d, but the number of random effects, including intercept, is %s" % (
                    gamma.shape[0],
                    num_random_features
                )

        num_features_to_generate = len(features_labels)
        # features covariance matrix describes covariances only presented in features_labels (meaningful features),
        # so we exclude the intercept feature when we generate random correlated data.
        if features_covariance_matrix is None:
            features_covariance_matrix = np.eye(num_features_to_generate)
        else:
            assert features_covariance_matrix.shape[0] == num_features_to_generate == \
                   features_covariance_matrix.shape[1], \
                "features_covariance_matrix should be n*n where n is the number of continuous features"

        data = {
            'fixed_features': [],
            'random_features': [],
            'categorical_features': [],
            'answers': [],
            'obs_stds': [],
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

        # Generate true group division based on active categorical features
        got_good_subdivision = False

        new_groups = []
        categorical_features_list = []

        all_data_in_one_matrix = None
        assert generator_params["num_attempts_to_generate_cat_features"] > 0, \
            "num_attempts_to_generate_cat_features should be > 0"
        for _ in range(generator_params["num_attempts_to_generate_cat_features"]):
            # generate ALL the data iteratively and
            # try to find features which don't give too much granular subdivision
            if num_features_to_generate > 0:
                all_data_in_one_matrix = np.random.multivariate_normal(np.zeros(num_features_to_generate),
                                                                       features_covariance_matrix,
                                                                       num_objects)
                # add intercept to the left
                all_data_in_one_matrix = np.hstack([np.ones(num_objects).reshape((-1, 1)), all_data_in_one_matrix])
            else:
                # just the intercept
                all_data_in_one_matrix = np.ones((num_objects, 1))
            groups_column = np.repeat(range(num_groups), groups_sizes).reshape((-1, 1))
            active_categorical_features = groups_column
            categorical_features = groups_column
            if num_categorical_features > 0:
                categorical_features = all_data_in_one_matrix[:, categorical_features_idx]
                categorical_features[categorical_features > 0] = 1
                categorical_features[categorical_features < 0] = 0
                all_data_in_one_matrix[:, categorical_features_idx] = categorical_features
                categorical_features = np.hstack([groups_column, categorical_features])
                if len(active_categorical_features_idx) > 0:
                    active_categorical_features = np.hstack(
                        [active_categorical_features, all_data_in_one_matrix[:, active_categorical_features_idx]])

            # add groups labels as an always-present categorical feature
            tupled_categorical_features = [tuple(s) for s in active_categorical_features]
            unique_sub_labels = set(tupled_categorical_features)
            sub_labels_counters = [tupled_categorical_features.count(s) for s in unique_sub_labels]
            if all([s > 2 for s in sub_labels_counters]):
                for s in unique_sub_labels:
                    subgroup_idxs = np.array([i for i, t in enumerate(tupled_categorical_features) if t == s])
                    new_groups.append((s, subgroup_idxs))
                    categorical_features_list.append(categorical_features[subgroup_idxs, :])
                got_good_subdivision = True
                break

        if not got_good_subdivision:
            raise Exception("No good subdivision found. Reduce the number of categorical features"
                            " or increase the number of objects")

        num_groups = len(new_groups)

        random_effects_list = []
        errors_list = []
        order_of_objects = []
        true_group_labels = np.zeros(num_objects)
        reference_loss_value = 0
        start = 0

        for i, (group_label, group_idxs) in enumerate(new_groups):
            group_size = len(group_idxs)

            group_continuous_features = all_data_in_one_matrix[np.ix_(group_idxs, continuous_features_idx)]
            group_fixed_features = group_continuous_features[:, fixed_features_idx]
            group_random_features = group_continuous_features[:, random_features_idx]

            random_effects = None
            if true_random_effects is not None:
                random_effects = dict(true_random_effects).get(group_label, None)
            if random_effects is None:
                random_effects = np.random.multivariate_normal(np.zeros(num_random_features), np.diag(gamma))

            if isinstance(obs_std, np.ndarray):
                if obs_std.shape[0] == sum(groups_sizes):
                    std = obs_std[start:group_size]
                elif obs_std.shape[0] == num_groups:
                    std = obs_std[i]
                else:
                    raise ValueError("len(obs_std) should be either num_groups or sum(groups_sizes)")

            elif isinstance(obs_std, (float, int)):
                std = obs_std
            else:
                raise ValueError("obs_std is not an array or int/float.")

            group_stds = std
            group_errors = np.random.randn(group_size) * std
            group_answers = group_fixed_features.dot(beta) + group_random_features.dot(random_effects) + group_errors

            true_group_labels[start: start + group_size] = i

            random_effects_list.append((group_label, random_effects))

            order_of_objects += list(range(start, start + group_size))
            start += group_size

            reference_loss_value += np.linalg.norm(group_errors) ** 2
            data['fixed_features'].append(group_fixed_features)
            data['random_features'].append(group_random_features)
            data['answers'].append(group_answers)
            data['obs_stds'].append(np.ones(group_size) * group_stds)

            errors_list.append(group_errors)

        data['group_labels'] = np.arange(start=0, stop=num_groups, step=1)
        data['order_of_objects'] = np.array(order_of_objects)
        data['categorical_features'] = categorical_features_list

        # save information about active categorical features
        active_categorical_set = [0] + [i + 1 for i, l in enumerate(categorical_features_idx) if
                                        features_labels[l - 1] == 5]

        # remove difference between active/inactive categorical features
        for i, label in enumerate(features_labels):
            if label == 6:
                features_labels[i] = 5

        #  [intercept] + [current_group_division, default_group_division] + [features] + [STDs]
        all_columns_labels = [3 if random_intercept else 1] + [0, 5] + features_labels + [4]
        data['column_labels'] = all_columns_labels

        # We pivot the groups back to the original group division
        generated_problem = LinearLMEProblem(**data).pivot(categorical_features_set=(0, ))
        generated_problem = generated_problem.to_x_y() if as_x_y else generated_problem

        if return_true_model_coefficients:
            # random_effects = np.array(random_effects_list)

            per_group_coefficients = get_per_group_coefficients(beta,
                                                                random_effects_list,
                                                                labels=np.array(all_columns_labels))
            true_parameters = {
                "beta": beta,
                "gamma": gamma,
                "per_group_coefficients": per_group_coefficients,
                "active_categorical_set": active_categorical_set,
                "true_group_labels": true_group_labels,
                "random_effects": random_effects_list,
                "errors": np.array(errors_list),
                "reference_loss_value": reference_loss_value
            }
            return generated_problem, true_parameters
        else:
            return generated_problem, None

    @staticmethod
    def from_x_y(x: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 columns_labels: List[int] = None,
                 random_intercept: bool = True,
                 add_group_as_categorical_feature=False,
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
            3 (both fixed and random), 4 (observation standard deviance), or 5 (categorical features).
            There should be only one 0 in the list. If columns_labels is None then it's assumed that
            it is the first row of x.

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
        assert set(columns_labels).issubset(
            (0, 1, 2, 3, 4, 5)), "Only 0, 1, 2, 3, 4, and 5 are allowed in columns_labels"
        assert len(columns_labels) == x.shape[1], "len(columns_labels) != x.shape[1] (not all columns are labelled)"
        # take the index of a column that stores group labels
        group_labels_idx = [i for i, label in enumerate(columns_labels) if label == 0]
        assert len(group_labels_idx) == 1, "There should be only one 0 in columns_labels"
        group_labels_idx = group_labels_idx[0]
        obs_std_idx = [i for i, label in enumerate(columns_labels) if label == 4]
        assert len(obs_std_idx) == 1, "There should be only one 4 in columns_labels"
        obs_std_idx = obs_std_idx[0]
        assert all(x[:, obs_std_idx] != 0), "Errors' STDs can't be zero. Check for zeros in the respective column."
        fixed_features_idx = [i for i, t in enumerate(columns_labels) if t == 1 or t == 3]
        random_features_idx = [i for i, t in enumerate(columns_labels) if t == 2 or t == 3]
        # We include the groups column to the list of categorical features
        categorical_features_idx = [group_labels_idx] if add_group_as_categorical_feature else []
        categorical_features_idx += [i for i, t in enumerate(columns_labels) if t == 5]
        num_categorical_features = len(categorical_features_idx)
        groups_labels = unique_labels(x[:, group_labels_idx])

        data = {
            'fixed_features': [],
            'random_features': [],
            'categorical_features': [] if num_categorical_features > 0 else None,
            'answers': None if y is None else [],
            'obs_stds': [],
            'group_labels': groups_labels,
            'column_labels': np.array([3 if random_intercept else 1] + columns_labels),
        }

        order_of_objects = []
        for label in groups_labels:
            objects_idx = x[:, group_labels_idx] == label
            order_of_objects += np.where(objects_idx)[0].tolist()
            fixed_features = x[np.ix_(objects_idx, fixed_features_idx)]
            # add an intercept column plus real features
            # noinspection PyTypeChecker
            data['fixed_features'].append(
                np.concatenate((np.ones((fixed_features.shape[0], 1)), fixed_features), axis=1))
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
            if num_categorical_features > 0:
                data['categorical_features'].append(x[np.ix_(objects_idx, categorical_features_idx)])

        data["order_of_objects"] = order_of_objects

        return LinearLMEProblem(**data)

    def to_x_y(self) -> Tuple[np.ndarray, np.ndarray]:
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
        if self.num_categorical_features > 0:
            all_categorical_features = np.concatenate(self.categorical_features, axis=0)

        all_stds = np.concatenate(self.obs_stds, axis=0)
        untitled_data = np.zeros((all_fixed_features.shape[0], len(self.column_labels) - 1))

        fixed_effects_counter = 1
        random_intercept = self.column_labels[0] == 3
        if random_intercept:
            random_effects_counter = 1
        else:
            random_effects_counter = 0
        categorical_features_counter = 0

        for i, label in enumerate(self.column_labels[1:]):
            if label == 0:
                untitled_data[:, i] = all_group_labels
            elif label == 1:
                untitled_data[:, i] = all_fixed_features[:, fixed_effects_counter]
                fixed_effects_counter += 1
            elif label == 2:
                untitled_data[:, i] = all_random_features[:, random_effects_counter]
                random_effects_counter += 1
            elif label == 3:
                untitled_data[:, i] = all_fixed_features[:, fixed_effects_counter]
                fixed_effects_counter += 1
                random_effects_counter += 1
            elif label == 4:
                untitled_data[:, i] = all_stds
            elif label == 5:
                untitled_data[:, i] = all_categorical_features[:, categorical_features_counter]
                categorical_features_counter += 1

        untitled_data = untitled_data[np.array(self.order_of_objects).argsort()]
        column_labels = np.array(self.column_labels[1:]).reshape((1, len(self.column_labels[1:])))
        data_with_column_labels = np.concatenate((column_labels, untitled_data), axis=0)
        if self.answers is not None:
            all_answers = np.concatenate(self.answers)
        else:
            all_answers = None
        all_answers = all_answers[np.array(self.order_of_objects).argsort()]
        return data_with_column_labels, all_answers

    def pivot(self, categorical_features_set):
        """
        Get a transformed problem such that its groups labels are formed by all different combinations of
        features values from categorical_features_set.

        Parameters
        ----------
        categorical_features_set : Set(Int), required
            Set of columns labels which will be used for pivoting

        Returns
        -------
        problem : LinearLMEProblem
            Pivoted problem
        """
        x, y = self.to_x_y()
        group_labels_idx = [i for i, label in enumerate(x[0, :]) if label == 0]
        assert len(group_labels_idx) == 1, "More than one group label column is found. Check labels."
        categorical_features_idxs = [i for i, label in enumerate(x[0, :]) if label == 5]
        indexing_features_idxs = np.array([categorical_features_idxs[i] for i in categorical_features_set])
        indexing_features = x[1:, indexing_features_idxs]
        tupled_indexing_features = [tuple(s) for s in indexing_features]
        for i, s in enumerate(set(tupled_indexing_features)):
            subgroup_idxs = np.array([i + 1 for i, t in enumerate(tupled_indexing_features) if t == s])
            x[subgroup_idxs, group_labels_idx[0]] = i
        return LinearLMEProblem.from_x_y(x, y, random_intercept=True if self.column_labels[0] == 3 else False)

    def reconfigure_columns(self, new_columns_labels):
        x, y = self.to_x_y()
        assert len(new_columns_labels) == x.shape[1], f"new_column_labels have size {len(new_columns_labels)}, " \
                                                      f"but X has {x.shape[1]} columns. These two should match."
        x[0, :] = new_columns_labels
        return LinearLMEProblem.from_x_y(x, y, random_intercept=True if self.column_labels[0] == 3 else 1)

    def bootstrap(self, seed=42, categorical_features_idx=None, do_bootstrap_objects=True):
        """
        Generate a bootstrap problem from this problem.

        Parameters
        ----------
        seed : int
            random seed
        categorical_features_idx
        do_bootstrap_objects

        Returns
        -------
        problem : LinearLMEProblem
            bootstrapped problem
        """
        np.random.seed(seed)
        if categorical_features_idx is None:
            categorical_features_idx = np.zeros(self.num_categorical_features, dtype=int)
            categorical_features_idx[1:] = np.random.choice(range(1, self.num_categorical_features),
                                                            size=self.num_categorical_features - 1,
                                                            replace=True)

        data = {
            'fixed_features': [],
            'random_features': [],
            'categorical_features': [] if self.num_categorical_features > 0 else None,
            'answers': None if self.answers is None else [],
            'obs_stds': [],
            'group_labels': self.group_labels,
            'column_labels': self.column_labels,
            'order_of_objects': [],
            'categorical_features_bootstrap_idx': categorical_features_idx,
        }

        for i, ((x, y, z, l), group_size) in enumerate(zip(self, self.groups_sizes)):
            if do_bootstrap_objects:
                objects_idx = np.random.choice(range(group_size), size=group_size, replace=True)
            else:
                objects_idx = np.array(range(group_size))
            data['fixed_features'].append(x[objects_idx, :])  # Continuous features are not bootstrapped
            # same for random effects
            data['random_features'].append(z[objects_idx, :])
            data['obs_stds'].append(l[objects_idx])
            if y is not None:
                data['answers'].append(y[objects_idx])
            data['order_of_objects'] += np.arange(group_size).tolist()

            if self.num_categorical_features > 0:
                data['categorical_features'].append(
                    self.categorical_features[i][np.ix_(objects_idx, categorical_features_idx)])

        return LinearLMEProblem(**data)


if __name__ == "__main__":
    problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[60, 40, 25],
                                                         features_labels=[3, 5, 3, 6, 2, 5],
                                                         random_intercept=False)

    bootstrap_problem = problem.bootstrap(seed=42)
    X1, y1 = problem.to_x_y()
    X2, y2 = bootstrap_problem.to_x_y()
    a = 3
    pass
