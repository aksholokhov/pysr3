import numpy as np
import pandas as pd
import time
import datetime

from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import SR3L1LmeModel, L1LmeModel
from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle

from tqdm import tqdm

if __name__ == "__main__":
    num_trials = 1

    model_parameters = {
        "lb": 40,
        "lg": 40,
        "initializer": "EM",
        "logger_keys": ('converged', 'loss',),
        "tol_oracle": 1e-3,
        "tol_solver": 1e-6,
        "max_iter_oracle": 1000,
        "max_iter_solver": 1000
    }

    cov = 0.8
    problem_parameters = {
        "groups_sizes": [20, 12, 14, 50, 11]*2,
        "features_labels": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]*2,
        "random_intercept": True,
        "obs_std": 0.1,
        "chance_missing": 0,
        "chance_outlier": 0.0,
        "outlier_multiplier": 5,
        "features_covariance_matrix": np.eye(20) + cov - np.eye(20)*cov
    }

    log = pd.DataFrame(columns=("i", "lambda", "model", "time", "mse", "evar", "loss",
                                "fe_tp", "fe_tn", "fe_fp", "fe_fn",
                                "re_tp", "re_tn", "re_fp", "re_fn",
                                "number_of_iterations", "converged"))
    try:
        for i in tqdm(range(num_trials)):
            seed = i
            np.random.seed(seed)
            true_beta = np.random.choice(2, size=21, p=np.array([0.5, 0.5]))
            if sum(true_beta) == 0:
                true_beta[0] = 1
            true_gamma = np.random.choice(2, size=21, p=np.array([0.3, 0.7])) * true_beta

            problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                       # beta=true_beta,
                                                                       # gamma=true_gamma,
                                                                       seed=seed)
            x, y = problem.to_x_y()

            oracle = LinearLMEOracle(problem)
            condition = oracle.get_condition_numbers()

            lam = 0.0
            all_coefficients_are_dead = False
            while not all_coefficients_are_dead:
                l1_converged = 1
                l1_SR3_converged = 1

                l1_model = L1LmeModel(**model_parameters,
                                      stepping="line-search",
                                      lam=lam)
                l1_SR3_model = SR3L1LmeModel(**model_parameters,
                                             stepping="fixed",
                                             lam=lam)
                tic = time.perf_counter()
                toc = tic
                try:
                    l1_model.fit_problem(problem)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    l1_converged = 0
                finally:
                    l1_y_pred = l1_model.predict_problem(problem)

                    l1_results = {
                        "i": i,
                        "lam": lam,
                        "model": "L1",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, l1_y_pred),
                        "evar": explained_variance_score(y, l1_y_pred),
                        "loss": l1_model.logger_.get("loss")[-1],
                        "fe_tp": np.mean((true_beta != 0) & (l1_model.coef_["beta"] != 0)),
                        "fe_tn": np.mean((true_beta == 0) & (l1_model.coef_["beta"] == 0)),
                        "fe_fp": np.mean((true_beta == 0) & (l1_model.coef_["beta"] != 0)),
                        "fe_fn": np.mean((true_beta != 0) & (l1_model.coef_["beta"] == 0)),
                        "re_tp": np.mean((true_gamma != 0) & (l1_model.coef_["gamma"] != 0)),
                        "re_tn": np.mean((true_gamma == 0) & (l1_model.coef_["gamma"] == 0)),
                        "re_fp": np.mean((true_gamma == 0) & (l1_model.coef_["gamma"] != 0)),
                        "re_fn": np.mean((true_gamma != 0) & (l1_model.coef_["gamma"] == 0)),
                        "number_of_iterations": len(l1_model.logger_.get("loss")),
                        "converged": l1_converged
                    }
                    log = log.append(l1_results, ignore_index=True)
                tic = time.perf_counter()
                toc = tic
                try:
                    l1_SR3_model.fit_problem(problem)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    l1_SR3_converged = 0
                finally:
                    l1_sr3_y_pred = l1_SR3_model.predict_problem(problem)

                    l1_sr3_results = {
                        "i": i,
                        "lam": lam,
                        "model": "SR3_L1",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, l1_sr3_y_pred),
                        "evar": explained_variance_score(y, l1_sr3_y_pred),
                        "loss": l1_SR3_model.logger_.get("loss")[-1],
                        "fe_tp": np.mean((true_beta != 0) & (l1_SR3_model.coef_["beta"] != 0)),
                        "fe_tn": np.mean((true_beta == 0) & (l1_SR3_model.coef_["beta"] == 0)),
                        "fe_fp": np.mean((true_beta == 0) & (l1_SR3_model.coef_["beta"] != 0)),
                        "fe_fn": np.mean((true_beta != 0) & (l1_SR3_model.coef_["beta"] == 0)),
                        "re_tp": np.mean((true_gamma != 0) & (l1_SR3_model.coef_["gamma"] != 0)),
                        "re_tn": np.mean((true_gamma == 0) & (l1_SR3_model.coef_["gamma"] == 0)),
                        "re_fp": np.mean((true_gamma == 0) & (l1_SR3_model.coef_["gamma"] != 0)),
                        "re_fn": np.mean((true_gamma != 0) & (l1_SR3_model.coef_["gamma"] == 0)),
                        "number_of_iterations": len(l1_SR3_model.logger_.get("loss")),
                        "converged": l1_SR3_converged
                    }
                    log = log.append(l1_sr3_results, ignore_index=True)

                    lam = 1.1 * (lam + 0.1)
                    print(f"lam={lam}, l1 fe = {sum(l1_model.coef_['beta'] != 0)}," +
                          f" l1 re = {sum(l1_model.coef_['gamma'] != 0)}, "+
                          f" sr3 fe = {sum(l1_SR3_model.coef_['beta'] != 0)}," +
                          f" sr3 re = {sum(l1_SR3_model.coef_['gamma'] != 0)}" )

                    if all(l1_model.coef_["beta"] == 0) and all(l1_model.coef_["gamma"] == 0) and all(
                            l1_SR3_model.coef_["beta"] == 0) and all(l1_SR3_model.coef_["gamma"] == 0):
                        all_coefficients_are_dead = True

    finally:
        now = datetime.datetime.now()
        log.to_csv(f"log_lasso_{now}.csv")
