import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import time
import datetime
from pathlib import Path
from matplotlib import ticker
from copy import deepcopy

from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import Sr3L0LmeModel, L0LmeModel, L1LmeModel, SR3L1LmeModel, CADLmeModel, SR3CADLmeModel
from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle

import pickle

from skmixed.helpers import random_effects_to_matrix

from tqdm import tqdm

from matplotlib import pyplot as plt

def initialize_models(model_parameters):
    model_parameters["fixed_step_len"] = 0.1 / 40
    model_parameters["lam"] = 1
    l0_sr3_model = Sr3L0LmeModel(**model_parameters,
                                 stepping="fixed")
    l1_sr3_model = SR3L1LmeModel(**model_parameters,
                                 stepping='fixed',
                                 )
    cad_sr3_model = SR3CADLmeModel(**model_parameters,
                                   stepping='fixed')
    model_parameters["fixed_step_len"] = 0.1 / 200
    model_parameters["lam"] = 10
    l1_model = L1LmeModel(**model_parameters,
                          stepping='fixed_max')

    cad_model = CADLmeModel(**model_parameters,
                            stepping='fixed_max')
    l0_model = L0LmeModel(**model_parameters,
                          stepping="fixed_max")


    return {#"L0": l0_model,
            #"L0_SR3": l0_sr3_model,
            "L1": l1_model,
            "L1_SR3": l1_sr3_model,
            "CAD": cad_model,
            "CAD_SR3": cad_sr3_model}


def assess(model, problem, y, true_model_parameters):
    tic = time.perf_counter()
    toc = tic
    converged = 1
    try:
        model.fit_problem(problem)
        toc = time.perf_counter()
    except np.linalg.LinAlgError:
        toc = time.perf_counter()
        converged = 0
    finally:
        y_pred = model.predict_problem(problem)
        true_beta = true_model_parameters["beta"]
        true_gamma = true_model_parameters["gamma"]

        results = {
            "time": toc - tic,
            "mse": mean_squared_error(y, y_pred),
            "evar": explained_variance_score(y, y_pred),
            "loss": model.logger_.get("loss")[-1],
            "fe_tp": np.sum((true_beta != 0) & (model.coef_["beta"] != 0)),
            "fe_tn": np.sum((true_beta == 0) & (model.coef_["beta"] == 0)),
            "fe_fp": np.sum((true_beta == 0) & (model.coef_["beta"] != 0)),
            "fe_fn": np.sum((true_beta != 0) & (model.coef_["beta"] == 0)),
            "re_tp": np.sum((true_gamma != 0) & (model.coef_["gamma"] != 0)),
            "re_tn": np.sum((true_gamma == 0) & (model.coef_["gamma"] == 0)),
            "re_fp": np.sum((true_gamma == 0) & (model.coef_["gamma"] != 0)),
            "re_fn": np.sum((true_gamma != 0) & (model.coef_["gamma"] == 0)),
            "number_of_iterations": len(model.logger_.get("loss")),
            "converged": converged
        }
        return results


def generate_noise_data(problem_parameters, model_parameters, noise_levels, true_beta, true_gamma, num_trials, logs_folder):
    model_parameters = deepcopy(model_parameters)
    log = pd.DataFrame(columns=("x", "trial", "model", "time", "mse", "evar", "loss",
                                "fe_tp", "fe_tn", "fe_fp", "fe_fn",
                                "re_tp", "re_tn", "re_fp", "re_fn",
                                "number_of_iterations", "converged"))

    for j, chance in tqdm(enumerate(noise_levels)):
        for i in tqdm(range(num_trials)):
            seed = 1000 * i + j
            np.random.seed(seed)

            problem_parameters["obs_std"] = chance
            model_parameters["rho"] = np.sqrt(chance)

            problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                       beta=true_beta,
                                                                       gamma=true_gamma,
                                                                       seed=seed)
            x, y = problem.to_x_y()
            for model_name, model in initialize_models(model_parameters).items():
                result = assess(model, problem, y, true_model_parameters)
                result["x"] = chance
                result["trial"] = i
                result["model"] = model_name
                log = log.append(result, ignore_index=True)
                tp = result['fe_tp'] + result['re_tp']
                fp = result['fe_fp'] + result['re_fp']
                fn = result['fe_fn'] + result['re_fn']
                print(
                    f"{model_name} done ({result['time']:.2f}): {result['number_of_iterations']}, {tp / (tp + 1 / 2 * (fn + fp)):.2f}")
    now = datetime.datetime.now()
    log.to_csv(logs_folder / f"log_cond_noise_{now}")
    print(f"logs saved to {logs_folder / f'log_cond_noise_{now}'}")
    problem_parameters["obs_std"] = 0.1


def plot_data(logs, figures_folder, title=""):
    models = ["L1", "CAD"]
    logs['tp'] = logs['fe_tp'] + logs['re_tp']
    logs['fp'] = logs['fe_fp'] + logs['re_fp']
    logs['fn'] = logs['fe_fn'] + logs['re_fn']
    logs['f1'] = logs['tp'] / (logs['tp'] + 1/2*(logs['fn'] + logs['fp']))
    logs['fe_f1'] = logs['fe_tp']  / (logs['fe_tp'] + 1/2*(logs['fe_fn'] + logs['fe_fp']))
    logs['re_f1'] = logs['re_tp'] / (logs['re_tp'] + 1 / 2 * (logs['re_fn'] + logs['re_fp']))
    logs['x'] = logs['x'].round(1)
    for model in models:

        data = logs[(logs["model"] == model) | (logs["model"] == model + "_SR3")]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x='x', y='fp', data=data, hue='model', ax=ax)
        #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:1.1f}'))
        ax.set_title(title)
        plt.show()


def generate_wide_matrix_data(problem_parameters, model_parameters, nums_covariates, num_trials, logs_folder, cov = 0.):
    #problem_parameters["groups_sizes"] = [10]*6
    model_parameters = deepcopy(model_parameters)

    log = pd.DataFrame(columns=("x", "trial", "model", "time", "mse", "evar", "loss",
                                "fe_tp", "fe_tn", "fe_fp", "fe_fn",
                                "re_tp", "re_tn", "re_fp", "re_fn",
                                "number_of_iterations", "converged"))

    for j, nc in tqdm(enumerate(nums_covariates)):
        problem_parameters["features_labels"] = [3] * nc
        problem_parameters["features_covariance_matrix"] = sp.linalg.block_diag(
            *([np.array([[1, cov],
                         [cov, 1]])] * int(nc / 2)))
        true_beta = np.array([1] + [1, 0] * int(nc / 2))
        true_gamma = np.array([1] + [1, 0] * int(nc / 2))

        for i in tqdm(range(num_trials)):
            seed = 1000 * i + j
            np.random.seed(seed)
            problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                       beta=true_beta,
                                                                       gamma=true_gamma,
                                                                       seed=seed)
            x, y = problem.to_x_y()
            for model_name, model in initialize_models(model_parameters).items():
                result = assess(model, problem, y, true_model_parameters)
                result["x"] = nc
                result["trial"] = i
                result["model"] = model_name
                log = log.append(result, ignore_index=True)
                tp = result['fe_tp'] + result['re_tp']
                fp = result['fe_fp'] + result['re_fp']
                fn = result['fe_fn'] + result['re_fn']
                print(
                    f"{model_name} done ({result['time']:.2f}): {result['number_of_iterations']}, {tp / (tp + 1 / 2 * (fn + fp)):.2f}")
    now = datetime.datetime.now()
    log.to_csv(logs_folder / f"log_many_covariates_{now}")
    print(f"logs saved to {logs_folder / f'log_many_covariates_{now}.csv'}")



def generate_missings_data(problem_parameters, model_parameters, chances_missing, true_beta, true_gamma, num_trials, logs_folder):

    model_parameters = deepcopy(model_parameters)

    log = pd.DataFrame(columns=("x", "trial", "model", "time", "mse", "evar", "loss",
                                        "fe_tp", "fe_tn", "fe_fp", "fe_fn",
                                        "re_tp", "re_tn", "re_fp", "re_fn",
                                        "number_of_iterations", "converged"))

    for j, chance in tqdm(enumerate(chances_missing)):
        for i in tqdm(range(num_trials)):
            seed = 1000 * i + j
            np.random.seed(seed)

            problem_parameters["chance_missing"] = chance

            problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                       beta=true_beta,
                                                                       gamma=true_gamma,
                                                                       seed=seed)
            x, y = problem.to_x_y()
            for model_name, model in initialize_models(model_parameters).items():
                result = assess(model, problem,  y, true_model_parameters)
                result["x"] = chance
                result["trial"] = i
                result["model"] = model_name
                log = log.append(result, ignore_index=True)
                tp = result['fe_tp'] + result['re_tp']
                fp = result['fe_fp'] + result['re_fp']
                fn = result['fe_fn'] + result['re_fn']
                print(f"{model_name} done ({result['time']:.2f}): {result['number_of_iterations']}, { tp / (tp + 1/2*(fn + fp)):.2f}")
    now = datetime.datetime.now()
    log.to_csv(logs_folder / f"log_cond_missing_{now}.csv")
    print(f"logs saved to {logs_folder / f'log_cond_missing_{now}.csv'}")
    problem_parameters["chance_missing"] = 0


if __name__ == "__main__":
    base_folder = Path("/Users/aksh/Storage/repos/skmixed/examples/paper_sum2021")

    logs_folder = base_folder / "logs"
    figures_folder = base_folder / "figures"
    tables_folder = base_folder / "tables"

    num_trials = 30
    num_covariates = 20

    model_parameters = {
        "lb": 20,
        "lg": 20,
        # "lb": 40,
        # "lg": 40,
        "initializer": "None",
        "logger_keys": ('converged', 'loss',),
        "tol_oracle": 1e-3,
        "tol_solver": 1e-3,
        "max_iter_oracle": 4,
        "max_iter_solver": 100000,
        "lam": 1,
        #"lam": 10,
        "rho": 0.33,
        "nnz_tbeta": 20,
        "nnz_tgamma": 20,
        #"fixed_step_len": 0.1 / 200,
        "fixed_step_len": 0.1/40,
        "warm_start": True,
    }
    correlation_between_adjacent_covariates = 0.0

    problem_parameters = {
        "groups_sizes": [20, 12, 14, 11]*2,
        "features_labels": [3] * num_covariates,
        "random_intercept": True,
        "obs_std": 0.1,
        "distribution": "uniform",
        "chance_missing": 0,
        "chance_outlier": 0.0,
        "outlier_multiplier": 10,
        "features_covariance_matrix": sp.linalg.block_diag(
            *([np.array([[1, correlation_between_adjacent_covariates],
                         [correlation_between_adjacent_covariates, 1]])] * int(num_covariates / 2)))
    }

    true_beta = np.array([1] + [1, 0] * int(num_covariates / 2))
    true_gamma = np.array([1] + [1, 0] * int(num_covariates / 2))

    chances_missing = np.arange(0, 0.4, 0.05)
    noise_levels = np.arange(0.1, 2, 0.1)
    nums_covariates = range(20, 60, 2)

    now = datetime.datetime.now()

    with open(Path(logs_folder) / f"params_conditioning_{now}.csv", 'wb') as f:
        pickle.dump({
            "num_trials": num_trials,
            "num_covariates": num_covariates,
            "model_parameters": model_parameters,
            "problem_parameters": problem_parameters,
            "chances_missing": chances_missing,
            "noise_levels": noise_levels,
            "nums_covariates": nums_covariates
        }, f)

    logs = pd.read_csv(logs_folder / "log_cond_noise_2021-06-16 14:01:35.934282")
    plot_data(logs, figures_folder, title="Noise level")

    #logs = pd.read_csv(logs_folder / "log_many_covariates_2021-06-17 02:20:29.555188")
    #plot_data(logs, figures_folder, title="Num features" )

    #generate_wide_matrix_data(problem_parameters, model_parameters, nums_covariates, num_trials, logs_folder)
    #generate_noise_data(problem_parameters, model_parameters, noise_levels, true_beta, true_gamma, num_trials,
    #                    logs_folder)