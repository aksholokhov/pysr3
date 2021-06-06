import numpy as np
import scipy as sp
import pandas as pd
import time
import datetime
import pickle
from pathlib import Path

from matplotlib import pyplot as plt
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

    cad_initials = {
        "beta": np.ones(num_covariates + 1),
        "gamma": np.ones(num_covariates + 1)
    }

    sr3_initials = {
        "beta": np.ones(num_covariates + 1),
        "gamma": np.ones(num_covariates + 1)
    }



def run_cad_comparison_experiment(num_trials, num_covariates, model_parameters, problem_parameters, cad_initials,
                                     sr3_initials, rho,
                                     logs_folder=".", figures_folder=".", tables_folder="."):

    log = pd.DataFrame(columns=("i", "rho", "model", "time", "mse", "evar", "loss",
                                "fe_tp", "fe_tn", "fe_fp", "fe_fn",
                                "re_tp", "re_tn", "re_fp", "re_fn",
                                "number_of_iterations", "converged"))

    try:
        for i in tqdm(range(num_trials)):
            seed = i
            np.random.seed(seed)
            true_beta = np.array([1] + [1, 0] * int(num_covariates / 2))
            true_gamma = np.array([1] + [1, 0] * int(num_covariates / 2))

            problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                       beta=true_beta,
                                                                       gamma=true_gamma,
                                                                       seed=seed)
            x, y = problem.to_x_y()
            lam = 0.0
            all_coefficients_are_dead = False


            while not (all_coefficients_are_dead or lam > 1e5):
                cad_converged = 1
                SR3_converged = 1

                CAD_model = CADLmeModel(**model_parameters,
                                        stepping="line-search",
                                        rho=rho,
                                        lam=lam)
                CAD_SR3_model = SR3CADLmeModel(**model_parameters,
                                               stepping="fixed",
                                               rho=rho,
                                               lam=lam)
                tic = time.perf_counter()
                toc = tic
                try:
                    CAD_model.fit_problem(problem, initial_parameters=cad_initials)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    cad_converged = 0
                finally:
                    CAD_y_pred = CAD_model.predict_problem(problem)

                    CAD_results = {
                        "i": i,
                        "lam": lam,
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
                        "converged": cad_converged
                    }
                    log = log.append(CAD_results, ignore_index=True)
                tic = time.perf_counter()
                toc = tic
                try:
                    CAD_SR3_model.fit_problem(problem, initial_parameters=sr3_initials)
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

                print(f"lam={lam}, ({CAD_results['time']:.2f}) cad fe = {sum(CAD_model.coef_['beta'] != 0)}," +
                      f" cad re = {sum(CAD_model.coef_['gamma'] != 0)}, " +
                      f" sr3 ({CAD_sr3_results['time']:.2f})  fe = {sum(CAD_SR3_model.coef_['beta'] != 0)}," +
                      f" sr3 re = {sum(CAD_SR3_model.coef_['gamma'] != 0)}")

                lam = 1.05 * (lam + 0.1)

                if rho < 1e-8:
                    all_coefficients_are_dead = True

    finally:
        now = datetime.datetime.now()
        log_filename = Path(logs_folder) / f"log_cad_{now}.csv"
        log.to_csv(log_filename)
        print(f"L1 experiment: data saved as {log_filename}")
        with open(Path(logs_folder) / f"params_cad_{now}.csv", 'wb') as f:
            pickle.dump({
                "num_trials": num_trials,
                "num_covariates": num_covariates,
                "model_parameters": model_parameters,
                "problem_parameters": problem_parameters,
                "rho": rho,
                "cad_initials": cad_initials,
                "sr3_initials": sr3_initials
            }, f)
    return log, now


def plot_cad_comparison(data, rho, suffix=None, figures_folder="."):
    data["tp"] = data["fe_tp"] + data["re_tp"]
    data["tn"] = data["fe_tn"] + data["re_tn"]
    data["fp"] = data["fe_fp"] + data["re_fp"]
    data["fn"] = data["fe_fn"] + data["re_fn"]

    data["fe_sensitivity"] = data["fe_tp"] / (data["fe_tp"] + data["fe_fn"])
    data["fe_specificity"] = data["fe_tn"] / (data["fe_tn"] + data["fe_fp"])
    data["fe_fpr"] = data["fe_fp"] / (data["fe_fp"] + data["fe_tn"])
    data["fe_f1"] = 2 * data["fe_tp"] / (2 * data["fe_tp"] + data["fe_fp"] + data["fe_fn"])
    data["fe_acc"] = (data["fe_tp"] + data["fe_tn"]) / (data["fe_tp"] + data["fe_fn"] + data["fe_tn"] + data["fe_fp"])

    data["re_sensitivity"] = data["re_tp"] / (data["re_tp"] + data["re_fn"])
    data["re_specificity"] = data["re_tn"] / (data["re_tn"] + data["re_fp"])
    data["re_fpr"] = data["re_fp"] / (data["re_fp"] + data["re_tn"])
    data["re_f1"] = 2 * data["re_tp"] / (2 * data["re_tp"] + data["re_fp"] + data["re_fn"])
    data["re_acc"] = (data["re_tp"] + data["re_tn"]) / (data["re_tp"] + data["re_fn"] + data["re_tn"] + data["re_fp"])

    data["sensitivity"] = data["tp"] / (data["tp"] + data["fn"])
    data["fpr"] = data["fp"] / (data["fp"] + data["tn"])
    data["f1"] = 2 * data["tp"] / (2 * data["tp"] + data["fp"] + data["fn"])
    data["acc"] = (data["tp"] + data["tn"]) / (data["tp"] + data["fn"] + data["tn"] + data["fp"])

    cad_data = data[data["model"] == "CAD"]
    sr3_cad_data = data[data["model"] == "SR3_CAD"]

    agg_data = sr3_cad_data.copy()
    agg_data = agg_data.merge(cad_data, on="lam", suffixes=("_sr3", "_cad"))

    base_size = 5
    fig = plt.figure(figsize=(2 * base_size, 2 * base_size))
    grid = plt.GridSpec(nrows=2, ncols=2)

    #     fe_plot = fig.add_subplot(grid[0, 2])
    #     fe_plot.scatter(agg_data["fe_fpr_sr3"], agg_data["fe_sensitivity_sr3"], label="sr3")
    #     fe_plot.scatter(agg_data["fe_fpr_l1"], agg_data["fe_sensitivity_l1"], label="l1")
    #     fe_plot.set_xlabel("FPR FE")
    #     fe_plot.set_ylabel("TPR FE")
    #     fe_plot.legend()

    #     re_plot = fig.add_subplot(grid[1, 2])
    #     re_plot.scatter(agg_data["re_fpr_sr3"], agg_data["re_sensitivity_sr3"], label="sr3")
    #     re_plot.scatter(agg_data["re_fpr_l1"], agg_data["re_sensitivity_l1"], label="l1")
    #     re_plot.set_xlabel("FPR RE")
    #     re_plot.set_ylabel("TPR RE")
    #     re_plot.legend()

    #     all_plot = fig.add_subplot(grid[0, 2])
    #     all_plot.scatter(agg_data["fpr_sr3"], agg_data["sensitivity_sr3"], label="sr3")
    #     all_plot.scatter(agg_data["fpr_l1"], agg_data["sensitivity_l1"], label="l1")
    #     all_plot.set_xlabel("FPR")
    #     all_plot.set_ylabel("TPR")
    #     all_plot.legend()

    fe_plot = fig.add_subplot(grid[0, :2])
    # fe_plot.semilogx(agg_data["lam"], agg_data["f1_l1"], label="F1 L1")
    fe_plot.semilogx(agg_data["lam"], agg_data["fe_f1_cad"], label="CAD")
    fe_plot.semilogx(agg_data["lam"], agg_data["fe_f1_sr3"], label="CAD SR3")
    fe_plot.legend(loc="lower left")
    fe_plot.set_xlabel(r"$\lambda$, strength of CAD regularizer")
    fe_plot.set_ylabel(r"F1, selection quality for fixed effects")
    fe_plot.set_title(f"Fixed-effects selection quality along CAD path, rho={rho}")

    re_plot = fig.add_subplot(grid[1, :2])
    # re_plot.semilogx(agg_data["lam"], agg_data["f1_sr3"], label="F1 SR3")
    re_plot.semilogx(agg_data["lam"], agg_data["re_f1_cad"], label="F1 for RE selection with CAD")
    re_plot.semilogx(agg_data["lam"], agg_data["re_f1_sr3"], label="F1 for RE selection with CAD SR3 ")
    re_plot.legend(loc="lower left")
    re_plot.set_xlabel(r"$\lambda$, strength of CAD regularizer")
    re_plot.set_ylabel(r"F1, selection quality for random effects")
    re_plot.set_title(f"Random-effects selection quality along CAD path, rho={rho}")

    #     lambda_l1_plot = fig.add_subplot(grid[3, :])
    #     lambda_l1_plot.semilogx(agg_data["lam"], agg_data["acc_l1"], label="Acc L1")
    #     lambda_l1_plot.semilogx(agg_data["lam"], agg_data["fe_acc_l1"], label="Acc FE L1")
    #     lambda_l1_plot.semilogx(agg_data["lam"], agg_data["re_acc_l1"], label="Acc RE L1")
    #     lambda_l1_plot.legend()

    #     lambda_sr3_plot = fig.add_subplot(grid[4, :])
    #     lambda_sr3_plot.semilogx(agg_data["lam"], agg_data["acc_sr3"], label="Acc SR3")
    #     lambda_sr3_plot.semilogx(agg_data["lam"], agg_data["fe_acc_sr3"], label="Acc FE SR3")
    #     lambda_sr3_plot.semilogx(agg_data["lam"], agg_data["re_acc_sr3"], label="Acc RE SR3")
    #     lambda_sr3_plot.legend()

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plot_filename = Path(figures_folder) / f"cad_comparison_{suffix if suffix else ''}.pdf"
    plt.savefig(plot_filename)
    print(f"CAD experiment: plot saved as {plot_filename}")