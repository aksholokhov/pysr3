import numpy as np

from lib.problems import LinearLMEProblem
from lib.old_solvers import LinearLMESolver


class LMEBootstrapper:
    def __init__(self, max_capacity: int = 100, seed: int = None):
        self.max_capacity = max_capacity
        self.seed = seed

    def __iter__(self):
        raise NotImplementedError("You need to pass a non-general bootstrapper, which inherits this abstract class")

    def __next__(self):
        raise NotImplementedError("You need to pass a non-general bootstrapper, which inherits this abstract class")


class NonParLinearLMEBootstrapper(LMEBootstrapper):
    def __init__(self, problem: LinearLMEProblem, max_capacity: int = 100, seed: int = None):
        super().__init__(max_capacity, seed)
        self.problem = problem

    def __iter__(self):
        self.random_generator = np.random.RandomState()
        if self.seed is None:
            seed = np.random.randint(0)
        else:
            seed = self.seed
        self.random_generator.seed(seed)
        self.__generator_counter = 0
        return self

    def __next__(self):
        if self.__generator_counter + 1 > self.max_capacity:
            raise StopIteration
        study_sizes = self.problem.study_sizes
        new_problem_data = {
            'features': [],
            'random_features': [],
            'answers': [],
            'observations_cov_matrices': []
        }
        for size, (x, y, z, l) in zip(study_sizes, self.problem):
            idx = self.random_generator.choice(size, size)
            new_problem_data['features'].append(x[idx, :])
            new_problem_data['random_features'].append(z[idx, :])
            new_problem_data['answers'].append(y[idx])
            new_problem_data['observations_cov_matrices'].append(np.diag(l[idx, idx]))
        self.__generator_counter += 1
        return LinearLMEProblem(**new_problem_data)


if __name__ == '__main__':
    noise_variance = 1e-1
    loss_tol = 1e-4
    max_iter = 1000

    random_seed = 46
    np.random.seed(random_seed)
    num_studies = np.random.randint(1, 5)
    study_sizes = np.random.randint(1, 100, num_studies)
    num_features = np.random.randint(2, min(min(study_sizes), 10))
    num_random_effects = np.random.randint(1, 5)
    problem, beta, gamma, us, errs = LinearLMEProblem.generate(study_sizes, noise_variance,
                                                               num_features=num_features,
                                                               num_random_effects=num_random_effects,
                                                               seed=random_seed)

    bootstrapper = NonParLinearLMEBootstrapper(problem, max_capacity=5, seed=random_seed)
    alg = LinearLMESolver()
    for sample_problem in bootstrapper:
        logger = alg.fit(sample_problem)
        print(logger['converged'], alg.beta)
    pass
