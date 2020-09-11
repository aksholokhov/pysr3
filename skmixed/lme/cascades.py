from typing import Set

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from skmixed.logger import Logger
from skmixed.lme.oracles import LinearLMEOracle


class LMECascadeOracle:
    def __init__(self, levels, hierarchy):
        self.levels = levels
        self.hierarchy = hierarchy
        self.oracles = []
        for level in levels:
            self.oracles.append(LinearLMEOracle(level))

    def loss(self, beta, gamma):
        loss = 0
        for level, oracle in enumerate(self.oracles):
            loss += oracle.loss(beta, (level+1)*gamma)
        return loss

    def gradient_gamma(self, beta, gamma):
        gradient = np.zeros(len(gamma))
        for level, oracle in enumerate(self.oracles):
            gradient += (level+1)*oracle.gradient_gamma(beta, (level+1)*gamma)
        return gradient

    def optimal_beta(self, gamma):
        total_kernel = 0
        total_tail = 0
        for level, oracle in enumerate(self.oracles):
            kernel, tail = oracle.optimal_beta((level+1)*gamma, _dont_solve_wrt_beta=True)
            total_kernel += kernel
            total_tail += tail
        return np.linalg.solve(total_kernel, total_tail)

    def optimal_random_effects(self, beta, gamma):
        random_effects = {1: np.zeros(len(gamma))}
        for level, oracle in enumerate(self.oracles):
            for i, group in enumerate(oracle.problem.group_labels):
                parent = self.hierarchy[group]
                u = random_effects[parent]
                beta_current = beta.copy()
                for k in range(len(beta)):
                    j = int(oracle.beta_to_gamma_map[k])
                    if j >= 0:
                        beta_current[k] += u[j]
                random_effects[group] = oracle.optimal_random_effects(beta_current, gamma)[i] + u
        return random_effects


class LMECascade(BaseEstimator, RegressorMixin):
    def __init__(self,
                 tol: float = 1e-4,
                 tol_inner: float = 1e-4,
                 solver: str = "pgd",
                 n_iter: int = 1000,
                 n_iter_inner: int = 20,
                 use_line_search: bool = True,
                 logger_keys: Set = ('converged', 'loss',)):
        self.tol = tol
        self.tol_inner = tol_inner
        self.solver = solver
        self.n_iter = n_iter
        self.n_iter_inner = n_iter_inner
        self.use_line_search = use_line_search
        self.logger_keys = logger_keys

    def fit_problem(self, levels, hierarchy):
        oracle = LMECascadeOracle(levels, hierarchy)

        num_levels = len(levels)
        num_fixed_effects = levels[0].num_fixed_effects
        num_random_effects = levels[0].num_random_effects

        beta = np.ones(num_fixed_effects)
        gamma = 2*np.ones(num_random_effects)

        def projected_direction(current_gamma: np.ndarray, current_direction: np.ndarray):
            proj_direction = current_direction.copy()
            for j, _ in enumerate(current_gamma):
                if current_gamma[j] <= 1e-15 and current_direction[j] <= 0:
                    proj_direction[j] = 0
            return proj_direction

        loss = oracle.loss(beta, gamma)
        self.logger_ = Logger(self.logger_keys)

        prev_beta = np.infty
        prev_gamma = np.infty
        iteration = 0

        while (np.linalg.norm(beta - prev_beta) > self.tol
                    or np.linalg.norm(gamma - prev_gamma) > self.tol) and iteration < self.n_iter:

                if iteration >= self.n_iter:
                    us = oracle.optimal_random_effects(beta, gamma)
                    if len(self.logger_keys) > 0:
                        self.logger_.log(**locals())
                    self.coef_ = {"beta": beta,
                                  "gamma": gamma,
                                  "random_effects": us
                                  }
                    self.logger_.add("converged", 0)
                    return self

                if self.solver == 'pgd':

                    prev_beta = beta
                    prev_gamma = gamma

                    beta = oracle.optimal_beta(gamma)

                    # =============== INNER (GAMMA) ITERATION ===========
                    gradient_gamma = oracle.gradient_gamma(beta, gamma)
                    # projecting the gradient to the set of constraints
                    direction = projected_direction(gamma, -gradient_gamma)

                    inner_iteration = 0
                    while (np.linalg.norm(direction) > self.tol_inner
                           and inner_iteration < self.n_iter_inner):
                        if self.use_line_search:
                            # line search method
                            step_len = 0.1
                            for i, _ in enumerate(gamma):
                                if direction[i] < 0:
                                    step_len = min(-gamma[i] / direction[i], step_len)

                            current_loss = oracle.loss(beta, gamma)

                            while (oracle.loss(beta, gamma + step_len * direction)
                                   >= (1 - np.sign(current_loss) * 1e-5) * current_loss):
                                step_len *= 0.5
                                if step_len <= 1e-15:
                                    break
                        else:
                            # fixed step size
                            step_len = 1 / iteration
                        if step_len <= 1e-15:
                            break
                        gamma = gamma + step_len * direction
                        gradient_gamma = oracle.gradient_gamma(beta, gamma)
                        direction = projected_direction(gamma, -gradient_gamma)
                        inner_iteration += 1

                iteration += 1
                loss = oracle.loss(beta, gamma)
                if len(self.logger_keys) > 0:
                    self.logger_.log(locals())

        us = oracle.optimal_random_effects(beta, gamma)

        self.logger_.add('converged', 1)
        # self.logger_.add('iterations', iteration)

        self.coef_ = {
            "beta": beta,
            "gamma": gamma,
            "random_effects": us,
        }

        return self


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    from numpy.linalg import lstsq
    import json

    from skmixed.lme.models import LinearLMESparseModel
    from skmixed.lme.problems import LinearLMEProblem
    from skmixed.lme.oracles import LinearLMEOracle

    dataset_generator_directory = Path("/Users/aksh/Storage/repos/multilevel-model")
    dataset = "simulation-2"
    dataset_path = dataset_generator_directory / dataset / (dataset + ".csv")
    hierarchy_path = dataset_generator_directory / "hierarchy.csv"
    true_coefficients_path = dataset_generator_directory / dataset / "KEY" / ("PARAMS-" + dataset + ".csv")
    true_answers_path = dataset_generator_directory / dataset / "KEY" / ("TRUTH-" + dataset + ".csv")
    true_parameters_path = dataset_generator_directory / dataset / ("SETTINGS-" + dataset + ".json")

    data = pd.read_csv(dataset_path)
    hierarchy = pd.read_csv(hierarchy_path)
    hierarchy = hierarchy[hierarchy.location_id.isin(data.location_id.unique())]
    hierarchy = hierarchy[['location_name', 'location_id', 'parent_id', 'level']].reset_index(drop=True)

    data = data.merge(hierarchy, on="location_id", how="left")

    levels = sorted(data["level"].unique())

    covariates_colnames = [column for column in data.columns if column.startswith("x_")]
    target_colname = ["observation"]

    num_fixed_effects = len(covariates_colnames) + 1

    data["intercept"] = 1
    data["predictions"] = 0
    data["residues"] = 0
    data["se"] = 0
    hierarchy["se"] = 1

    hierarchy_dict = {row["location_id"]: row["parent_id"] for i, row in hierarchy.iterrows()}

    current_level = data["level"] == 0
    level0_data = data[current_level][["intercept"] + covariates_colnames].to_numpy()
    level0_target = data[current_level]["observation"].to_numpy()
    level0_coefficients = lstsq(level0_data, level0_target, rcond=None)[0]
    data.loc[current_level, "predictions"] = level0_data.dot(level0_coefficients)
    data.loc[current_level, "residues"] = data.loc[current_level, "observation"] - data.loc[
        current_level, "predictions"]
    current_se = np.sqrt((data.loc[current_level, "residues"].var()))

    problems = []

    for level in levels:
        current_level = data["level"] == level
        data.loc[current_level, "se"] = current_se
        X = data[current_level][covariates_colnames + ["location_id", "se"]].to_numpy()
        column_labels = [1] * len(covariates_colnames) + [0, 4]
        X = np.vstack([column_labels, X])
        y = data[current_level]["observation"].to_numpy()
        problem = LinearLMEProblem.from_x_y(X, y, random_intercept=True)
        problems.append(problem)

    model = LMECascade()
    model.fit_problem(problems, hierarchy_dict)
    beta = model.coef_["beta"]

    answers = [{"location_id": group, "intercept": beta[0] + u[0], "effect_0": beta[1]} for group, u in
               model.coef_["random_effects"].items()]
    coefficients = pd.DataFrame(answers)
    true_coefficients = pd.read_csv(true_coefficients_path)
    coefficients = coefficients.merge(true_coefficients, on="location_id", how="inner", suffixes=('_pred', '_true'))
    with open(true_parameters_path) as f:
        true_parameters = json.load(f)

    coefficients[["location_id", "intercept_true", "intercept_pred", "effect_0_true", "effect_0_pred"]]