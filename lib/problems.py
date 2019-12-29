import numpy as np


class LMEProblem:
    def __init__(self):
        pass


class LinearLMEProblem(LMEProblem):

    def __init__(self, features, random_features, observations_cov_matrices, answers=None):
        super(LinearLMEProblem, self).__init__()

        assert len(features) == len(answers) == len(random_features) == len(
            observations_cov_matrices), "num of studies is inconsistent"

        self.features = features
        self.answers = answers
        self.random_features = random_features
        self.observations_cov_matrices = observations_cov_matrices

        self.num_random_effects = random_features[0].shape[1]
        self.num_features = features[0].shape[1]
        self.study_sizes = [x.shape[0] for x in features]
        self.num_studies = len(self.study_sizes)
        self.num_obs = sum(self.study_sizes)

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
            return self.features[j], answers, self.random_features[j], self.observations_cov_matrices[j]
        else:
            raise StopIteration

    # TODO: fix obs_std name
    @staticmethod
    def generate(num_studies=None, study_sizes=None, obs_std=0.1, beta=None, gamma=None, true_random_effects=None,
                 num_features=None, num_random_effects=None,
                 intercept=True, seed=None, return_true_parameters=True):

        if seed is not None:
            np.random.seed(seed)

        if num_studies is None:
            if study_sizes is None:
                num_studies = np.random.randint(2, 8)
            else:
                num_studies = len(study_sizes)

        if study_sizes is None:
            study_sizes = np.random.randint(3, 100, num_studies)

        # dimensions
        num_obs = np.sum(study_sizes)
        num_studies = len(study_sizes)

        if beta is None and num_features is not None:
            beta = np.random.rand(num_features)
        elif beta is not None and num_features is None:
            num_features = beta.size
        elif beta is None and num_features is None:
            num_features = np.random.randint(2, min(study_sizes))
            beta = np.random.rand(num_features)
        else:
            assert len(beta) == num_features, "len(beta) and num_features don't match"

        if gamma is None and num_random_effects is not None:
            gamma = np.random.rand(
                num_random_effects) + 0.1  # the problem is unstable when gamma is toooo small TODO: figure out why
        elif gamma is not None and num_random_effects is None:
            num_random_effects = gamma.size
        elif gamma is None and num_random_effects is None:
            num_random_effects = np.random.randint(1, 5)
            gamma = np.random.rand(num_random_effects) + 0.1  # same
        else:
            assert len(gamma) == num_random_effects, "len(gamma) and num_random_effects don't match"

        # create data
        features = np.random.rand(num_obs, num_features)

        if intercept:
            features[:, 0] = 1.0

        if num_random_effects == 1:
            random_features = np.ones((num_obs, 1))
        else:
            random_features = np.random.rand(num_obs, num_random_effects)

        random_effects_by_study = []
        random_effects_impact = np.zeros(num_obs)
        start = 0
        for j, size in enumerate(study_sizes):
            end = start + size
            if true_random_effects is None:
                random_effects = np.random.randn(num_random_effects) * np.sqrt(gamma)
            else:
                random_effects = true_random_effects[j]
            random_effects_by_study.append(random_effects)
            random_effects_impact[start:end] = random_features[start:end, :].dot(random_effects)
            start += size

        random_effects_by_study = np.array(random_effects_by_study)

        measurements_errors = np.random.randn(num_obs)

        if isinstance(obs_std, (float, int)) or len(obs_std) == num_obs:
            measurements_errors *= obs_std
        elif len(obs_std) == num_studies:
            measurements_errors *= np.repeat(obs_std, study_sizes)
        else:
            raise AssertionError("wrong size of obs_std")

        answers = features.dot(beta) + random_effects_impact + measurements_errors
        measurements_errors_by_study = []
        data = {
            'observations_cov_matrices': [],
            'features': [],
            'random_features': [],
            'answers': []
        }

        start = 0
        for i, size in enumerate(study_sizes):
            end = start + size
            data['features'].append(features[start:end])
            data['random_features'].append(random_features[start:end])
            measurements_errors_by_study.append(measurements_errors[start:end])
            data['answers'].append(answers[start:end])
            if isinstance(obs_std, (float, int)):
                data['observations_cov_matrices'].append(np.eye(size) * obs_std)
            elif len(obs_std) == num_obs:
                data['observations_cov_matrices'].append(np.diag(obs_std[start:end]))
            else:
                data['observations_cov_matrices'].append(np.eye(size) * obs_std[i])
            start += size

        if return_true_parameters:
            return LinearLMEProblem(**data), beta, gamma, random_effects_by_study, measurements_errors_by_study
        else:
            return LinearLMEProblem(**data)

    # def __str__(self):
    #     output = "LinearLMEProblem \n"
    #     output += " Beta: " + ' '.join([str(t) for t in self.__beta]) + '\n'
    #     output += " Gamma: " + ' '.join([str(t) for t in self.__gamma]) + '\n'
    #     output += " Random effects: \n "
    #     output += '\n '.join([str(t) for t in self.__random_effects])
    #     output += '\n ' + 'Study sizes: %s' % self.study_sizes
    #     return output


if __name__ == '__main__':
    problem = LinearLMEProblem.generate(study_sizes=[10, 30, 50], obs_std=1, num_features=3, num_random_effects=2)
    pass
