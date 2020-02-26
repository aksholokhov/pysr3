from typing import Union, Sized, List, Optional, Tuple, Any

import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y


class LMEProblem(object):
    def __init__(self, **kwargs):
        pass

    def from_x_y(self, **kwargs):
        pass

    def to_x_y(self, **kwargs):
        pass


class LinearLMEProblem(LMEProblem):

    def __init__(self,
                 features: List[np.ndarray],
                 random_features: List[np.ndarray],
                 obs_stds: Union[int, float, np.ndarray],
                 group_labels: np.ndarray,
                 column_labels: List[Tuple[int, int]],
                 order_of_objects: np.ndarray,
                 answers=None):
        super(LinearLMEProblem, self).__init__()

        self.features = features
        self.answers = answers
        self.random_features = random_features
        self.obs_stds = obs_stds

        self.num_random_effects = random_features[0].shape[1]
        self.num_features = features[0].shape[1]
        self.study_sizes = [x.shape[0] for x in features]
        self.num_studies = len(self.study_sizes)
        self.num_obs = sum(self.study_sizes)
        self.group_labels = group_labels
        self.column_labels = column_labels
        self.order_of_objects = order_of_objects

    def __iter__(self):
        self.__iteration_pos = 0
        return self

    def __next__(self):
        j = self.__iteration_pos
        if j < len(self.features):
            self.__iteration_pos += 1
            if self.answers is None:
                answers = None
            else:
                answers = self.answers[j]
            return self.features[j], answers, self.random_features[j], self.obs_stds[j]
        else:
            raise StopIteration

    @staticmethod
    def generate(num_studies: Optional[int] = None,
                 study_sizes: Optional[List[int]] = None,
                 num_fixed_effects: Optional[int] = None,
                 num_random_effects: Optional[int] = None,
                 both_fixed_and_random_effects: Optional[np.ndarray] = None,
                 features_covariance_matrix=None,
                 random_features_covariance_matrix=None,
                 obs_std: Optional[Union[int, float, Sized]] = 0.1,
                 beta: Optional[np.ndarray] = None,
                 gamma: Optional[np.ndarray] = None,
                 true_random_effects: Optional[np.ndarray] = None,
                 seed: int = None,
                 return_true_parameters: bool = True,
                 **kwargs
                 ) -> Tuple[Any, Optional[dict]]:
        """
        Generates a random mixed-effects problem with given parameters

        Y_i = X_i*beta + Z*u_i + e_i, where

        u_i ~ N(0, diag(gamma)),  e_i ~ N(0, diag(obs_std)), and i is from 1 to num_studies.

        Parameters
        ----------
        both_fixed_and_random_effects
        random_features_covariance_matrix
        features_covariance_matrix
        num_studies: int
                number of studies (a.k.a. clusters or groups). Defaults to None,
                in which case it's generated randomly from U[2, 8).
        study_sizes: List[int], shape=[num_studies]
                List of study sizes. Defaults to None,
                in which case it's generated randomly from U[3, 100)*num_studies.
        num_fixed_effects: int
                number of fixed effects (size of beta). Defaults to None, INCLUDES INTERCEPT
                in which case it's len(beta) if beta != None, otherwise generated randomly from U[2, min(study_sizes)).
        num_random_effects: int
                number of random effects (size of gamma). Defaults to None, INCLUDES INTERCEPT
                in which case it's len(gamma) if gamma != None, otherwise generated randomly from U[2, num_features].
        obs_std: {float, np.ndarray}, shape = 1 or [num_studies] or [sum(study_sizes)]
                variance of observation errors
        beta:
        gamma:
        true_random_effects:
        seed:
        return_true_parameters:

        Returns
        -------
        problem:
                Generated problem

        """
        if seed is not None:
            np.random.seed(seed)

        if num_studies is None:
            if study_sizes is None:
                num_studies = np.random.randint(2, 8)
            else:
                num_studies = len(study_sizes)

        if study_sizes is None:
            study_sizes = np.random.randint(3, 100, num_studies)

        num_studies = len(study_sizes)

        if beta is None and num_fixed_effects is not None:
            beta = np.random.rand(num_fixed_effects)
        elif beta is not None and num_fixed_effects is None:
            num_fixed_effects = beta.size
        elif beta is None and num_fixed_effects is None:
            num_fixed_effects = np.random.randint(2, min(study_sizes))
            beta = np.random.rand(num_fixed_effects)
        else:
            assert len(beta) == num_fixed_effects, "len(beta) and num_features don't match"

        if gamma is None and num_random_effects is not None:
            gamma = np.random.rand(num_random_effects)
        elif gamma is not None and num_random_effects is None:
            num_random_effects = gamma.size
        elif gamma is None and num_random_effects is None:
            num_random_effects = np.random.randint(2, num_fixed_effects + 1)
            gamma = np.random.rand(num_random_effects) + 0.1  # same
        else:
            assert len(gamma) == num_random_effects, "len(gamma) and num_random_effects don't match"

        if both_fixed_and_random_effects is None:
            num_both_fixed_and_random_effects = np.random.randint(0, min(num_fixed_effects, num_random_effects))
            both_fixed_and_random_effects = np.random.randint(0,
                                                              num_fixed_effects,
                                                              num_both_fixed_and_random_effects)
        else:
            assert np.all(both_fixed_and_random_effects[:-1] <= both_fixed_and_random_effects[1:]), \
                "both_fixed_and_random_effects should be strictly ascending"
            num_both_fixed_and_random_effects = len(both_fixed_and_random_effects)

        data = {
            'features': [],
            'random_features': [],
            'answers': [],
            'obs_stds': [],
        }

        if features_covariance_matrix is None:
            features_covariance_matrix = np.eye(num_fixed_effects)
        if random_features_covariance_matrix is None:
            random_features_covariance_matrix = np.eye(num_random_effects)

        random_effects_list = []
        errors_list = []
        order_of_objects = []
        start = 0
        for i, size in enumerate(study_sizes):
            features = np.random.multivariate_normal(np.zeros(num_fixed_effects), features_covariance_matrix, size)
            features[:, 0] = 1  # the first feature is always the intercept
            order_of_objects += list(range(start, start + size))
            start += size
            random_features = np.random.multivariate_normal(np.zeros(num_random_effects),
                                                            random_features_covariance_matrix, size)
            random_features[:, 0] = 1
            random_features[:, :num_both_fixed_and_random_effects] = features[:, both_fixed_and_random_effects]

            if true_random_effects is not None:
                random_effects = true_random_effects[i]
            else:
                random_effects = np.random.multivariate_normal(np.zeros(num_random_effects), np.diag(gamma))
            if isinstance(obs_std, list) and len(obs_std) == num_studies:
                std = obs_std[i]
            elif isinstance(obs_std, (float, int)):
                std = obs_std
            else:
                raise ValueError("obs_std is not a list or int/float.")
            errors = np.random.randn(size) * std
            answers = features.dot(beta) + random_features.dot(random_effects) + errors

            data['features'].append(features)
            data['random_features'].append(random_features)
            data['answers'].append(answers)
            data['obs_stds'].append(np.ones(size) * std)
            random_effects_list.append(random_effects)
            errors_list.append(errors)

        data['group_labels'] = np.arange(start=0, stop=num_studies, step=1)
        data['order_of_objects'] = np.array(order_of_objects)
        data['column_labels'] = [0]
        column_labels = [1]*num_fixed_effects
        for t in both_fixed_and_random_effects:
            column_labels[t] = 3
        column_labels += [2]*(num_random_effects - num_both_fixed_and_random_effects)
        column_labels = column_labels + [4, 0]
        data['column_labels'] = column_labels
        if return_true_parameters:
            true_parameters = {
                "beta": beta,
                "gamma": gamma,
                "random_effects": np.array(random_effects_list),
                "errors": np.array(errors_list)
            }
            return LinearLMEProblem(**data), true_parameters
        else:
            return LinearLMEProblem(**data), None

    @staticmethod
    def from_x_y(x: np.ndarray, y: Optional[np.ndarray] = None, columns_labels: List[int] = None,
                 random_intercept: bool = True, **kwargs):
        """
        Transforms matrices x (data) and y(answers) into an instance of LinearLMEProblem

        Parameters
        ----------
        random_intercept
        x: array-like, shape = [m,n]
            Data.
        y: array-like, shape = [m]
            Answers.
        columns_labels: List, shape = [n], Optional
            A list of column labels which can be 0 (group labels), 1 (fixed effect), 2 (random effect),
             3 (both fixed and random), or 4 (observation standard deviance). There should be only one 0 in the list.
             If it's None then it's assumed that it is the first row of x.
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
            'features': [],
            'random_features': [],
            'answers': [],
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
            data['features'].append(np.concatenate((np.ones((features.shape[0], 1)), features), axis=1))
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
        all_features = np.concatenate(self.features, axis=0)
        all_random_features = np.concatenate(self.random_features, axis=0)
        all_stds = np.concatenate(self.obs_stds, axis=0)
        untitled_data = np.zeros((all_features.shape[0], len(self.column_labels)-1))

        fixed_effects_counter = 1
        random_intercept = self.column_labels[0]
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


if __name__ == '__main__':
    problem = LinearLMEProblem.generate(study_sizes=[10, 30, 50], obs_std=1, num_fixed_effects=3, num_random_effects=2)
    pass
