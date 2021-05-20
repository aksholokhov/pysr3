import numpy as np
import scipy as sp
import pandas as pd
import time
import datetime

from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import SR3L1LmeModel, L1LmeModel, L0LmeModel, Sr3L0LmeModel
from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle

from tqdm import tqdm

if __name__ == "__main__":
    num_trials = 1

    num_covariates = 40

    model_parameters = {
        "lb": 40,
        "lg": 40,
        "initializer": "None",
        "logger_keys": ('converged', 'loss',),
        "tol_oracle": 1e-3,
        "tol_solver": 1e-6,
        "max_iter_oracle": 10,
        "max_iter_solver": 10000,
    }

    cov = 0.0
    problem_parameters = {
        "groups_sizes": [20, 12, 14, 50, 11] * 2,
        "features_labels": [3] * num_covariates,
        "random_intercept": True,
        "obs_std": 0.1,
        "chance_missing": 0,
        "chance_outlier": 0.0,
        "outlier_multiplier": 5,
        # "features_covariance_matrix": np.eye(num_covariates)
        "features_covariance_matrix": sp.linalg.block_diag(
            *([np.array([[1, cov], [cov, 1]])] * int(num_covariates / 2)))
    }

    l0_initials = {
        "beta": np.ones(num_covariates + 1),
        "gamma": np.ones(num_covariates + 1)
    }

    SR3_initials = {
        "beta": np.ones(num_covariates + 1),
        "gamma": np.ones(num_covariates + 1)
    }

    log = pd.DataFrame(columns=("i", "nnz", "model", "time", "mse", "evar", "loss",
                                "fe_tp", "fe_tn", "fe_fp", "fe_fn",
                                "re_tp", "re_tn", "re_fp", "re_fn",
                                "number_of_iterations", "converged"))

    try:
        for i in tqdm(range(num_trials)):
            l0_converged = 1
            l0_SR3_converged = 1
            seed = i
            np.random.seed(seed)
            # true_beta = np.random.choice(2, size=num_covariates+1, p=np.array([0.5, 0.5]))
            # if sum(true_beta) == 0:
            #     true_beta[0] = 1
            # true_gamma = np.random.choice(2, size=num_covariates+1, p=np.array([0.3, 0.7])) * true_beta
            true_beta = np.array([1] + [1, 0] * int(num_covariates / 2))
            true_gamma = np.array([1] + [1, 0] * int(num_covariates / 2))

            problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                       beta=true_beta,
                                                                       gamma=true_gamma,
                                                                       seed=seed)
            x, y = problem.to_x_y()

            oracle = LinearLMEOracle(problem)
            condition = oracle.get_condition_numbers()

            for j in range(problem.num_fixed_effects):

                l0_model = L0LmeModel(**model_parameters,
                                      stepping="line-search",
                                      nnz_tbeta=j,
                                      nnz_tgamma=j)
                l0_SR3_model = Sr3L0LmeModel(**model_parameters,
                                             stepping="fixed",
                                             nnz_tbeta=j,
                                             nnz_tgamma=j)
                tic = time.perf_counter()
                toc = tic
                try:
                    l0_model.fit_problem(problem, initial_parameters=l0_initials)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    l0_converged = 0
                finally:
                    l0_y_pred = l0_model.predict_problem(problem)

                    l0_results = {
                        "i": i,
                        "nnz": j,
                        "model": "L0",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, l0_y_pred),
                        "evar": explained_variance_score(y, l0_y_pred),
                        "loss": l0_model.logger_.get("loss")[-1],
                        "fe_tp": np.sum((true_beta != 0) & (l0_model.coef_["beta"] != 0)),
                        "fe_tn": np.sum((true_beta == 0) & (l0_model.coef_["beta"] == 0)),
                        "fe_fp": np.sum((true_beta == 0) & (l0_model.coef_["beta"] != 0)),
                        "fe_fn": np.sum((true_beta != 0) & (l0_model.coef_["beta"] == 0)),
                        "re_tp": np.sum((true_gamma != 0) & (l0_model.coef_["gamma"] != 0)),
                        "re_tn": np.sum((true_gamma == 0) & (l0_model.coef_["gamma"] == 0)),
                        "re_fp": np.sum((true_gamma == 0) & (l0_model.coef_["gamma"] != 0)),
                        "re_fn": np.sum((true_gamma != 0) & (l0_model.coef_["gamma"] == 0)),
                        "number_of_iterations": len(l0_model.logger_.get("loss")),
                        "converged": l0_converged
                    }
                    log = log.append(l0_results, ignore_index=True)
                tic = time.perf_counter()
                toc = tic
                try:
                    l0_SR3_model.fit_problem(problem, initial_parameters=SR3_initials)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    l0_SR3_converged = 0
                finally:
                    l0_sr3_y_pred = l0_SR3_model.predict_problem(problem)

                    l0_sr3_results = {
                        "i": i,
                        "nnz": j,
                        "model": "SR3_L0",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, l0_sr3_y_pred),
                        "evar": explained_variance_score(y, l0_sr3_y_pred),
                        "loss": l0_SR3_model.logger_.get("loss")[-1],
                        "fe_tp": np.sum((true_beta != 0) & (l0_SR3_model.coef_["beta"] != 0)),
                        "fe_tn": np.sum((true_beta == 0) & (l0_SR3_model.coef_["beta"] == 0)),
                        "fe_fp": np.sum((true_beta == 0) & (l0_SR3_model.coef_["beta"] != 0)),
                        "fe_fn": np.sum((true_beta != 0) & (l0_SR3_model.coef_["beta"] == 0)),
                        "re_tp": np.sum((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] != 0)),
                        "re_tn": np.sum((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] == 0)),
                        "re_fp": np.sum((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] != 0)),
                        "re_fn": np.sum((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] == 0)),
                        "number_of_iterations": len(l0_SR3_model.logger_.get("loss")),
                        "converged": l0_SR3_converged
                    }
                    log = log.append(l0_sr3_results, ignore_index=True)

                print(f"nnz={j}, l1 fe = {sum(l0_model.coef_['beta'] != 0)}," +
                      f" l1 re = {sum(l0_model.coef_['gamma'] != 0)}, " +
                      f" sr3 fe = {sum(l0_SR3_model.coef_['beta'] != 0)}," +
                      f" sr3 re = {sum(l0_SR3_model.coef_['gamma'] != 0)}")

    finally:
        now = datetime.datetime.now()
        log.to_csv(f"log_l0_{now}.csv")
