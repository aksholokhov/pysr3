import numpy as np

from lib.problems import LinearLMEProblem
from lib.oracles import LinearLMEOracle, LinearLMEOracleRegularized
from lib.solvers import LinearLMERegSolver

if __name__ == '__main__':
    random_seed = 213
    study_sizes = [20]*3
    test_study_sizes = [10]
    num_features = 6
    num_random_effects = 6
    obs_std = 1e-1
    method = "VariableProjectionGD"
    initializer = "EM"
    lb = 0
    lg = 1
    how_close = 0
    tol = 1e-5
    max_iter = 100

    beta = np.ones(num_features)
    gamma = np.ones(num_random_effects)

    # beta is [0, 1, 1, 1 ...]
    # gamma is [0, 1, 1, ..., 1, 0]
    #beta[0] = 0
    #gamma[0] = 0
    #gamma[-1] = 0

    train, beta, gamma, random_effects, errs = LinearLMEProblem.generate(study_sizes=study_sizes,
                                                                         num_features=num_features,
                                                                         beta=beta,
                                                                         gamma=gamma,
                                                                         num_random_effects=num_random_effects,
                                                                         how_close_z_to_x=how_close,
                                                                         obs_std=obs_std,
                                                                         seed=random_seed)

    empirical_gamma = np.sum(random_effects ** 2, axis=0) / len(study_sizes)

    # these are oracle and method which are capable of performing feature selection
    train_oracle = LinearLMEOracleRegularized(train, lb=lb, lg=lg, mode='naive')

    pred_beta = train_oracle.optimal_beta_reg(gamma, np.zeros(num_features))
    pass