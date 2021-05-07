import numpy as np
import scipy as sp
import pandas as pd
import time
import datetime

from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import Sr3L0LmeModel, L0LmeModel
from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle

import pickle


from skmixed.helpers import random_effects_to_matrix

from tqdm import tqdm

from matplotlib import pyplot as plt

if __name__ == "__main__":

    num_trials = 5
    num_covariates = 40

    model_parameters = {
        "lb": 40,
        "lg": 40,
        "initializer": "None",
        "logger_keys": ('converged', 'loss',),
        "tol_oracle": 1e-3,
        "tol_solver": 1e-5,
        "max_iter_oracle": 10000,
        "max_iter_solver": 10000,
        "warm_start": True,
    }

    problem_parameters = {
        "groups_sizes": [20, 12, 14, 50, 11]*2,
        "features_labels": [3]*num_covariates,
        "random_intercept": True,
        "obs_std": 0.3,
        "chance_missing": 0,
        "chance_outlier": 0.0,
        "outlier_multiplier": 10
    }



    # chances_missing = np.arange(0, 0.7, 0.05)
    # log = pd.DataFrame(columns=("j", "i", "chance", "model", "time", "mse", "evar", "loss",
    #                             "fe_tp", "fe_tn", "fe_fp", "fe_fn",
    #                             "re_tp", "re_tn", "re_fp", "re_fn",
    #                             "number_of_iterations", "converged"))
    # try:
    #     for j, chance in tqdm(enumerate(chances_missing)):
    #         for i in tqdm(range(num_trials)):
    #
    #             seed = 1000*i
    #             np.random.seed(seed)
    #
    #             problem_parameters["chance_missing"] = chance
    #
    #             true_beta = np.random.choice(2, size=num_covariates+1, p=np.array([0.5, 0.5]))
    #             if sum(true_beta) == 0:
    #                 true_beta[0] = 1
    #             true_gamma = np.random.choice(2, size=num_covariates+1, p=np.array([0.3, 0.7])) * true_beta
    #
    #             problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
    #                                                                        beta=true_beta,
    #                                                                        gamma=true_gamma,
    #                                                                        seed=seed)
    #             x, y = problem.to_x_y()
    #
    #             oracle = LinearLMEOracle(problem)
    #             condition = oracle.get_condition_numbers()
    #
    #             l0_converged=1
    #             l0_SR3_converged=1
    #
    #             l0_model = L0LmeModel(**model_parameters,
    #                                   stepping="line-search",
    #                                   nnz_tbeta=sum(true_beta),
    #                                   nnz_tgamma=sum(true_gamma))
    #             l0_SR3_model = Sr3L0LmeModel(**model_parameters,
    #                                          stepping="fixed",
    #                                          nnz_tbeta=sum(true_beta),
    #                                          nnz_tgamma=sum(true_gamma))
    #             tic = time.perf_counter()
    #             toc = tic
    #             try:
    #                 l0_model.fit_problem(problem)
    #                 toc = time.perf_counter()
    #             except np.linalg.LinAlgError:
    #                 toc = time.perf_counter()
    #                 l0_converged=0
    #             finally:
    #                 l0_y_pred = l0_model.predict_problem(problem)
    #
    #                 l0_results = {
    #                     "j": j,
    #                     "i": i,
    #                     "chance": chance,
    #                     "model": "L0",
    #                     "time": toc - tic,
    #                     "mse": mean_squared_error(y, l0_y_pred),
    #                     "evar": explained_variance_score(y, l0_y_pred),
    #                     "loss": l0_model.logger_.get("loss")[-1],
    #                     "fe_tp": np.mean((true_beta != 0) & (l0_model.coef_["beta"] != 0)),
    #                     "fe_tn": np.mean((true_beta == 0) & (l0_model.coef_["beta"] == 0)),
    #                     "fe_fp": np.mean((true_beta == 0) & (l0_model.coef_["beta"] != 0)),
    #                     "fe_fn": np.mean((true_beta != 0) & (l0_model.coef_["beta"] == 0)),
    #                     "re_tp": np.mean((true_gamma != 0) & (l0_model.coef_["gamma"] != 0)),
    #                     "re_tn": np.mean((true_gamma == 0) & (l0_model.coef_["gamma"] == 0)),
    #                     "re_fp": np.mean((true_gamma == 0) & (l0_model.coef_["gamma"] != 0)),
    #                     "re_fn": np.mean((true_gamma != 0) & (l0_model.coef_["gamma"] == 0)),
    #                     "number_of_iterations": len(l0_model.logger_.get("loss")),
    #                     "converged": l0_converged
    #                 }
    #                 log = log.append(l0_results, ignore_index=True)
    #             tic = time.perf_counter()
    #             toc = tic
    #             try:
    #                 l0_SR3_model.fit_problem(problem)
    #                 toc = time.perf_counter()
    #             except np.linalg.LinAlgError:
    #                 toc = time.perf_counter()
    #                 l0_SR3_converged = 0
    #             finally:
    #                 l0_sr3_y_pred = l0_SR3_model.predict_problem(problem)
    #
    #                 l0_sr3_results = {
    #                     "j": j,
    #                     "i": i,
    #                     "chance": chance,
    #                     "model": "SR3_L0",
    #                     "time": toc - tic,
    #                     "mse": mean_squared_error(y, l0_sr3_y_pred),
    #                     "evar": explained_variance_score(y, l0_sr3_y_pred),
    #                     "loss": l0_SR3_model.logger_.get("loss")[-1],
    #                     "fe_tp": np.mean((true_beta != 0) & (l0_SR3_model.coef_["beta"] != 0)),
    #                     "fe_tn": np.mean((true_beta == 0) & (l0_SR3_model.coef_["beta"] == 0)),
    #                     "fe_fp": np.mean((true_beta == 0) & (l0_SR3_model.coef_["beta"] != 0)),
    #                     "fe_fn": np.mean((true_beta != 0) & (l0_SR3_model.coef_["beta"] == 0)),
    #                     "re_tp": np.mean((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] != 0)),
    #                     "re_tn": np.mean((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] == 0)),
    #                     "re_fp": np.mean((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] != 0)),
    #                     "re_fn": np.mean((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] == 0)),
    #                     "number_of_iterations": len(l0_SR3_model.logger_.get("loss")),
    #                     "converged": l0_SR3_converged
    #                 }
    #                 log = log.append(l0_sr3_results, ignore_index=True)
    # finally:
    #     now = datetime.datetime.now()
    #     log.to_csv(f"log_missing_{now}.csv")
    #
    # problem_parameters["chance_missing"] = 0

    #
    # chances_outlier = np.arange(0, 0.9, 0.05)
    #
    # log = pd.DataFrame(columns=("j", "i", "chance", "model", "time", "mse", "evar", "loss",
    #                             "fe_tp", "fe_tn", "fe_fp", "fe_fn",
    #                             "re_tp", "re_tn", "re_fp", "re_fn",
    #                             "number_of_iterations", "converged"))
    # try:
    #     for j, chance in tqdm(enumerate(chances_outlier)):
    #         for i in tqdm(range(num_trials)):
    #
    #             seed = 1000*i
    #             np.random.seed(seed)
    #
    #             problem_parameters["chance_outlier"] = chance
    #
    #             true_beta = np.random.choice(2, size=num_covariates+1, p=np.array([0.5, 0.5]))
    #             if sum(true_beta) == 0:
    #                 true_beta[0] = 1
    #             true_gamma = np.random.choice(2, size=num_covariates+1, p=np.array([0.3, 0.7])) * true_beta
    #
    #             problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
    #                                                                        beta=true_beta,
    #                                                                        gamma=true_gamma,
    #                                                                        seed=seed)
    #             x, y = problem.to_x_y()
    #
    #             oracle = LinearLMEOracle(problem)
    #             condition = oracle.get_condition_numbers()
    #
    #             l0_converged = 1
    #             l0_SR3_converged = 1
    #
    #             l0_model = L0LmeModel(**model_parameters,
    #                                   stepping="line-search",
    #                                   nnz_tbeta=sum(true_beta),
    #                                   nnz_tgamma=sum(true_gamma))
    #             l0_SR3_model = Sr3L0LmeModel(**model_parameters,
    #                                          stepping="fixed",
    #                                          nnz_tbeta=sum(true_beta),
    #                                          nnz_tgamma=sum(true_gamma))
    #             tic = time.perf_counter()
    #             toc = tic
    #             try:
    #                 l0_model.fit_problem(problem)
    #                 toc = time.perf_counter()
    #             except np.linalg.LinAlgError:
    #                 toc = time.perf_counter()
    #                 l0_converged = 0
    #             finally:
    #                 l0_y_pred = l0_model.predict_problem(problem)
    #
    #                 l0_results = {
    #                     "j": j,
    #                     "i": i,
    #                     "chance": chance,
    #                     "model": "L0",
    #                     "time": toc - tic,
    #                     "mse": mean_squared_error(y, l0_y_pred),
    #                     "evar": explained_variance_score(y, l0_y_pred),
    #                     "loss": l0_model.logger_.get("loss")[-1],
    #                     "fe_tp": np.mean((true_beta != 0) & (l0_model.coef_["beta"] != 0)),
    #                     "fe_tn": np.mean((true_beta == 0) & (l0_model.coef_["beta"] == 0)),
    #                     "fe_fp": np.mean((true_beta == 0) & (l0_model.coef_["beta"] != 0)),
    #                     "fe_fn": np.mean((true_beta != 0) & (l0_model.coef_["beta"] == 0)),
    #                     "re_tp": np.mean((true_gamma != 0) & (l0_model.coef_["gamma"] != 0)),
    #                     "re_tn": np.mean((true_gamma == 0) & (l0_model.coef_["gamma"] == 0)),
    #                     "re_fp": np.mean((true_gamma == 0) & (l0_model.coef_["gamma"] != 0)),
    #                     "re_fn": np.mean((true_gamma != 0) & (l0_model.coef_["gamma"] == 0)),
    #                     "number_of_iterations": len(l0_model.logger_.get("loss")),
    #                     "converged": l0_converged
    #                 }
    #                 log = log.append(l0_results, ignore_index=True)
    #             tic = time.perf_counter()
    #             toc = tic
    #             try:
    #                 l0_SR3_model.fit_problem(problem)
    #                 toc = time.perf_counter()
    #             except np.linalg.LinAlgError:
    #                 toc = time.perf_counter()
    #                 l0_SR3_converged = 0
    #             finally:
    #                 l0_sr3_y_pred = l0_SR3_model.predict_problem(problem)
    #
    #                 l0_sr3_results = {
    #                     "j": j,
    #                     "i": i,
    #                     "chance": chance,
    #                     "model": "SR3_L0",
    #                     "time": toc - tic,
    #                     "mse": mean_squared_error(y, l0_sr3_y_pred),
    #                     "evar": explained_variance_score(y, l0_sr3_y_pred),
    #                     "loss": l0_SR3_model.logger_.get("loss")[-1],
    #                     "fe_tp": np.mean((true_beta != 0) & (l0_SR3_model.coef_["beta"] != 0)),
    #                     "fe_tn": np.mean((true_beta == 0) & (l0_SR3_model.coef_["beta"] == 0)),
    #                     "fe_fp": np.mean((true_beta == 0) & (l0_SR3_model.coef_["beta"] != 0)),
    #                     "fe_fn": np.mean((true_beta != 0) & (l0_SR3_model.coef_["beta"] == 0)),
    #                     "re_tp": np.mean((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] != 0)),
    #                     "re_tn": np.mean((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] == 0)),
    #                     "re_fp": np.mean((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] != 0)),
    #                     "re_fn": np.mean((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] == 0)),
    #                     "number_of_iterations": len(l0_SR3_model.logger_.get("loss")),
    #                     "converged": l0_SR3_converged
    #                 }
    #                 log = log.append(l0_sr3_results, ignore_index=True)
    # finally:
    #     now = datetime.datetime.now()
    #     log.to_csv(f"log_outliers_{now}.csv")
    # problem_parameters["chance_outlier"] = 0
    #
    #
    # correlations = np.arange(0, 1.0, 0.05)
    # log = pd.DataFrame(columns=("j", "i", "chance", "model", "time", "mse", "evar", "loss",
    #                             "fe_tp", "fe_tn", "fe_fp", "fe_fn",
    #                             "re_tp", "re_tn", "re_fp", "re_fn",
    #                             "number_of_iterations", "converged"))
    # try:
    #     for j, chance in tqdm(enumerate(correlations)):
    #         for i in tqdm(range(num_trials)):
    #
    #             seed = 1000*i
    #             np.random.seed(seed)
    #
    #             true_beta = np.array([1] + [1, 0]*int(num_covariates/2))
    #             true_gamma = np.array([1] + [1, 0, 0, 0]*int(num_covariates/4))
    #             problem_parameters["features_covariance_matrix"] = sp.linalg.block_diag(*([np.array([[1, chance], [chance, 1]])]*int(num_covariates/2)))
    #
    #
    #             problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
    #                                                                        beta=true_beta,
    #                                                                        gamma=true_gamma,
    #                                                                        seed=seed)
    #             x, y = problem.to_x_y()
    #
    #             oracle = LinearLMEOracle(problem)
    #             condition = oracle.get_condition_numbers()
    #
    #             l0_converged = 1
    #             l0_SR3_converged = 1
    #
    #             l0_model = L0LmeModel(**model_parameters,
    #                                   stepping="line-search",
    #                                   nnz_tbeta=sum(true_beta),
    #                                   nnz_tgamma=sum(true_gamma))
    #             l0_SR3_model = Sr3L0LmeModel(**model_parameters,
    #                                          stepping="fixed",
    #                                          nnz_tbeta=sum(true_beta),
    #                                          nnz_tgamma=sum(true_gamma))
    #             tic = time.perf_counter()
    #             toc = tic
    #             try:
    #                 l0_model.fit_problem(problem)
    #                 toc = time.perf_counter()
    #                 print(f"L0 done: {toc - tic}")
    #             except np.linalg.LinAlgError:
    #                 toc = time.perf_counter()
    #                 l0_converged = 0
    #             finally:
    #                 l0_y_pred = l0_model.predict_problem(problem)
    #
    #                 l0_results = {
    #                     "j": j,
    #                     "i": i,
    #                     "chance": chance,
    #                     "model": "L0",
    #                     "time": toc - tic,
    #                     "mse": mean_squared_error(y, l0_y_pred),
    #                     "evar": explained_variance_score(y, l0_y_pred),
    #                     "loss": l0_model.logger_.get("loss")[-1],
    #                     "fe_tp": np.mean((true_beta != 0) & (l0_model.coef_["beta"] != 0)),
    #                     "fe_tn": np.mean((true_beta == 0) & (l0_model.coef_["beta"] == 0)),
    #                     "fe_fp": np.mean((true_beta == 0) & (l0_model.coef_["beta"] != 0)),
    #                     "fe_fn": np.mean((true_beta != 0) & (l0_model.coef_["beta"] == 0)),
    #                     "re_tp": np.mean((true_gamma != 0) & (l0_model.coef_["gamma"] != 0)),
    #                     "re_tn": np.mean((true_gamma == 0) & (l0_model.coef_["gamma"] == 0)),
    #                     "re_fp": np.mean((true_gamma == 0) & (l0_model.coef_["gamma"] != 0)),
    #                     "re_fn": np.mean((true_gamma != 0) & (l0_model.coef_["gamma"] == 0)),
    #                     "number_of_iterations": len(l0_model.logger_.get("loss")),
    #                     "converged": l0_converged
    #                 }
    #                 log = log.append(l0_results, ignore_index=True)
    #             tic = time.perf_counter()
    #             toc = tic
    #             try:
    #                 l0_SR3_model.fit_problem(problem)
    #                 toc = time.perf_counter()
    #                 print(f"SR3 done: {toc - tic}")
    #             except np.linalg.LinAlgError:
    #                 toc = time.perf_counter()
    #                 l0_SR3_converged = 0
    #             finally:
    #                 l0_sr3_y_pred = l0_SR3_model.predict_problem(problem)
    #
    #                 l0_sr3_results = {
    #                     "j": j,
    #                     "i": i,
    #                     "chance": chance,
    #                     "model": "SR3_L0",
    #                     "time": toc - tic,
    #                     "mse": mean_squared_error(y, l0_sr3_y_pred),
    #                     "evar": explained_variance_score(y, l0_sr3_y_pred),
    #                     "loss": l0_SR3_model.logger_.get("loss")[-1],
    #                     "fe_tp": np.mean((true_beta != 0) & (l0_SR3_model.coef_["beta"] != 0)),
    #                     "fe_tn": np.mean((true_beta == 0) & (l0_SR3_model.coef_["beta"] == 0)),
    #                     "fe_fp": np.mean((true_beta == 0) & (l0_SR3_model.coef_["beta"] != 0)),
    #                     "fe_fn": np.mean((true_beta != 0) & (l0_SR3_model.coef_["beta"] == 0)),
    #                     "re_tp": np.mean((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] != 0)),
    #                     "re_tn": np.mean((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] == 0)),
    #                     "re_fp": np.mean((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] != 0)),
    #                     "re_fn": np.mean((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] == 0)),
    #                     "number_of_iterations": len(l0_SR3_model.logger_.get("loss")),
    #                     "converged": l0_SR3_converged
    #                 }
    #                 log = log.append(l0_sr3_results, ignore_index=True)
    # finally:
    #     now = datetime.datetime.now()
    #     log.to_csv(f"log_correlation_{now}.csv")
    #
    # problem_parameters["features_covariance_matrix"] = np.eye(num_covariates)



    noise_amplitude = np.arange(0.1, 4, 0.05)

    log = pd.DataFrame(columns=("j", "i", "chance", "model", "time", "mse", "evar", "loss",
                                "fe_tp", "fe_tn", "fe_fp", "fe_fn",
                                "re_tp", "re_tn", "re_fp", "re_fn",
                                "number_of_iterations", "converged"))
    try:
        for j, chance in tqdm(enumerate(noise_amplitude)):
            for i in tqdm(range(num_trials)):

                seed = 2000*i
                np.random.seed(seed)

                problem_parameters["obs_std"] = chance

                true_beta = np.random.choice(2, size=num_covariates+1, p=np.array([0.5, 0.5]))
                if sum(true_beta) == 0:
                    true_beta[0] = 1
                true_gamma = np.random.choice(2, size=num_covariates+1, p=np.array([0.3, 0.7])) * true_beta

                problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                           beta=true_beta,
                                                                           gamma=true_gamma,
                                                                           seed=seed)
                x, y = problem.to_x_y()

                oracle = LinearLMEOracle(problem)
                condition = oracle.get_condition_numbers()

                l0_converged = 1
                l0_SR3_converged = 1

                l0_model = L0LmeModel(**model_parameters,
                                      stepping="line-search",
                                      nnz_tbeta=sum(true_beta),
                                      nnz_tgamma=sum(true_gamma))
                l0_SR3_model = Sr3L0LmeModel(**model_parameters,
                                             stepping="fixed",
                                             nnz_tbeta=sum(true_beta),
                                             nnz_tgamma=sum(true_gamma))
                tic = time.perf_counter()
                toc = tic
                try:
                    l0_model.fit_problem(problem)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    l0_converged = 0
                finally:
                    l0_y_pred = l0_model.predict_problem(problem)

                    l0_results = {
                        "j": j,
                        "i": i,
                        "chance": chance,
                        "model": "L0",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, l0_y_pred),
                        "evar": explained_variance_score(y, l0_y_pred),
                        "loss": l0_model.logger_.get("loss")[-1],
                        "fe_tp": np.mean((true_beta != 0) & (l0_model.coef_["beta"] != 0)),
                        "fe_tn": np.mean((true_beta == 0) & (l0_model.coef_["beta"] == 0)),
                        "fe_fp": np.mean((true_beta == 0) & (l0_model.coef_["beta"] != 0)),
                        "fe_fn": np.mean((true_beta != 0) & (l0_model.coef_["beta"] == 0)),
                        "re_tp": np.mean((true_gamma != 0) & (l0_model.coef_["gamma"] != 0)),
                        "re_tn": np.mean((true_gamma == 0) & (l0_model.coef_["gamma"] == 0)),
                        "re_fp": np.mean((true_gamma == 0) & (l0_model.coef_["gamma"] != 0)),
                        "re_fn": np.mean((true_gamma != 0) & (l0_model.coef_["gamma"] == 0)),
                        "number_of_iterations": len(l0_model.logger_.get("loss")),
                        "converged": l0_converged
                    }
                    log = log.append(l0_results, ignore_index=True)
                    print(f"L0 done: {toc - tic}")
                tic = time.perf_counter()
                toc = tic
                try:
                    l0_SR3_model.fit_problem(problem)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    l0_SR3_converged = 0
                finally:
                    l0_sr3_y_pred = l0_SR3_model.predict_problem(problem)

                    l0_sr3_results = {
                        "j": j,
                        "i": i,
                        "chance": chance,
                        "model": "SR3_L0",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, l0_sr3_y_pred),
                        "evar": explained_variance_score(y, l0_sr3_y_pred),
                        "loss": l0_SR3_model.logger_.get("loss")[-1],
                        "fe_tp": np.mean((true_beta != 0) & (l0_SR3_model.coef_["beta"] != 0)),
                        "fe_tn": np.mean((true_beta == 0) & (l0_SR3_model.coef_["beta"] == 0)),
                        "fe_fp": np.mean((true_beta == 0) & (l0_SR3_model.coef_["beta"] != 0)),
                        "fe_fn": np.mean((true_beta != 0) & (l0_SR3_model.coef_["beta"] == 0)),
                        "re_tp": np.mean((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] != 0)),
                        "re_tn": np.mean((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] == 0)),
                        "re_fp": np.mean((true_gamma == 0) & (l0_SR3_model.coef_["gamma"] != 0)),
                        "re_fn": np.mean((true_gamma != 0) & (l0_SR3_model.coef_["gamma"] == 0)),
                        "number_of_iterations": len(l0_SR3_model.logger_.get("loss")),
                        "converged": l0_SR3_converged
                    }
                    log = log.append(l0_sr3_results, ignore_index=True)
                    print(f"SR3 done: {toc - tic}")
    finally:
        now = datetime.datetime.now()
        log.to_csv(f"log_noise_{now}.csv")

    problem_parameters["obs_std"] = 0