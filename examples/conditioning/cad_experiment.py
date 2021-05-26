import numpy as np
import scipy as sp
import pandas as pd
import time
import datetime

from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import CADLmeModel, SR3CADLmeModel
from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle

from tqdm import tqdm

if __name__ == "__main__":
    num_trials = 1

    num_covariates = 20

    model_parameters = {
        "lb": 40,
        "lg": 40,
        "initializer": "None",
        "logger_keys": ('converged', 'loss',),
        "tol_oracle": 1e-3,
        "tol_solver": 1e-6,
        "max_iter_oracle": 10000,
        "max_iter_solver": 10000
    }

    cov = 0.0
    problem_parameters = {
        "groups_sizes": [20, 12, 14, 50, 11]*2,
        "features_labels": [3]*num_covariates,
        "random_intercept": True,
        "obs_std": 0.1,
        "chance_missing": 0,
        "chance_outlier": 0.0,
        "outlier_multiplier": 5,
        # "features_covariance_matrix": np.eye(num_covariates)
        "features_covariance_matrix": sp.linalg.block_diag(*([np.array([[1, cov], [cov, 1]])]*int(num_covariates/2)))
    }

    log = pd.DataFrame(columns=("i", "rho", "model", "time", "mse", "evar", "loss",
                                "fe_tp", "fe_tn", "fe_fp", "fe_fn",
                                "re_tp", "re_tn", "re_fp", "re_fn",
                                "number_of_iterations", "converged"))

    try:
        for i in tqdm(range(num_trials)):
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

            rho = 1e8
            all_coefficients_are_dead = False

            CAD_initials = {
                "beta": np.ones(num_covariates + 1),
                "gamma": np.ones(num_covariates + 1)
            }

            SR3_initials = {
                "beta": np.ones(num_covariates + 1),
                "gamma": np.ones(num_covariates + 1)
            }

            while not all_coefficients_are_dead:
                CAD_converged = 1
                SR3_converged = 1

                CAD_model = CADLmeModel(**model_parameters,
                                        stepping="line-search",
                                        rho=rho)
                CAD_SR3_model = SR3CADLmeModel(**model_parameters,
                                               stepping="fixed",
                                               rho=rho)
                tic = time.perf_counter()
                toc = tic
                try:
                    CAD_model.fit_problem(problem, initial_parameters=CAD_initials)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    CAD_converged = 0
                finally:
                    CAD_y_pred = CAD_model.predict_problem(problem)

                    CAD_results = {
                        "i": i,
                        "lam": rho,
                        "model": "CAD",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, CAD_y_pred),
                        "evar": explained_variance_score(y, CAD_y_pred),
                        "loss": CAD_model.logger_.get("loss")[-1],
                        "fe_tp": np.sum((true_beta != 0) & (CAD_model.coef_["beta"] != 0)),
                        "fe_tn": np.sum((true_beta == 0) & (CAD_model.coef_["beta"] == 0)),
                        "fe_fp": np.sum((true_beta == 0) & (CAD_model.coef_["beta"] != 0)),
                        "fe_fn": np.sum((true_beta != 0) & (CAD_model.coef_["beta"] == 0)),
                        "re_tp": np.sum((true_gamma != 0) & (CAD_model.coef_["gamma"] != 0)),
                        "re_tn": np.sum((true_gamma == 0) & (CAD_model.coef_["gamma"] == 0)),
                        "re_fp": np.sum((true_gamma == 0) & (CAD_model.coef_["gamma"] != 0)),
                        "re_fn": np.sum((true_gamma != 0) & (CAD_model.coef_["gamma"] == 0)),
                        "number_of_iterations": len(CAD_model.logger_.get("loss")),
                        "converged": CAD_converged
                    }
                    log = log.append(CAD_results, ignore_index=True)
                    # l1_initials["beta"] = l1_model.coef_["beta"]
                    # l1_initials["gamma"] = l1_model.coef_["gamma"]
                tic = time.perf_counter()
                toc = tic
                try:
                    CAD_SR3_model.fit_problem(problem, initial_parameters=SR3_initials)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    SR3_converged = 0
                finally:
                    CAD_sr3_y_pred = CAD_SR3_model.predict_problem(problem)

                    CAD_sr3_results = {
                        "i": i,
                        "lam": rho,
                        "model": "SR3_CAD",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, CAD_sr3_y_pred),
                        "evar": explained_variance_score(y, CAD_sr3_y_pred),
                        "loss": CAD_SR3_model.logger_.get("loss")[-1],
                        "fe_tp": np.sum((true_beta != 0) & (CAD_SR3_model.coef_["beta"] != 0)),
                        "fe_tn": np.sum((true_beta == 0) & (CAD_SR3_model.coef_["beta"] == 0)),
                        "fe_fp": np.sum((true_beta == 0) & (CAD_SR3_model.coef_["beta"] != 0)),
                        "fe_fn": np.sum((true_beta != 0) & (CAD_SR3_model.coef_["beta"] == 0)),
                        "re_tp": np.sum((true_gamma != 0) & (CAD_SR3_model.coef_["gamma"] != 0)),
                        "re_tn": np.sum((true_gamma == 0) & (CAD_SR3_model.coef_["gamma"] == 0)),
                        "re_fp": np.sum((true_gamma == 0) & (CAD_SR3_model.coef_["gamma"] != 0)),
                        "re_fn": np.sum((true_gamma != 0) & (CAD_SR3_model.coef_["gamma"] == 0)),
                        "number_of_iterations": len(CAD_SR3_model.logger_.get("loss")),
                        "converged": SR3_converged
                    }
                    log = log.append(CAD_sr3_results, ignore_index=True)
                    # SR3_initials["beta"] = l1_SR3_model.coef_["beta"]
                    # SR3_initials["gamma"] = l1_SR3_model.coef_["gamma"]

                print(f"rho={rho}, cad fe = {sum(CAD_model.coef_['beta'] != 0)}," +
                      f" cad re = {sum(CAD_model.coef_['gamma'] != 0)}, " +
                      f" sr3 fe = {sum(CAD_SR3_model.coef_['beta'] != 0)}," +
                      f" sr3 re = {sum(CAD_SR3_model.coef_['gamma'] != 0)}")

                rho = rho / 2

                if rho < 1e-8:
                    all_coefficients_are_dead = True

    finally:
        now = datetime.datetime.now()
        log.to_csv(f"log_cad_{now}.csv")
