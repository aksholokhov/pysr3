import numpy as np
import pandas as pd
import time
import datetime
import pickle

from matplotlib import pyplot as plt

from pathlib import Path

from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import SR3L1LmeModel, L1LmeModel, L0LmeModel, Sr3L0LmeModel
from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle

from tqdm import tqdm


def run_l0_comparison_experiment(num_trials, num_covariates, model_parameters, problem_parameters, l0_initials, sr3_initials,
                                 logs_folder=".", figures_folder=".", tables_folder="."):

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
                    l0_SR3_model.fit_problem(problem, initial_parameters=sr3_initials)
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
        log_filename = Path(logs_folder) / f"log_l0_{now}.csv"
        log.to_csv(log_filename)
        print(f"L0 experiment: data saved as {log_filename}")
        with open(Path(logs_folder) / f"params_l0_{now}.csv", 'wb') as f:
            pickle.dump({
                "num_trials": num_trials,
                "num_covariates": num_covariates,
                "model_parameters": model_parameters,
                "problem_parameters": problem_parameters,
                "l0_initials": l0_initials,
                "sr3_initials": sr3_initials
            }, f)
    return log, f"{now}"


def plot_l0_comparison(data, suffix=None, figures_folder="."):
    data["nnz"] = data["nnz"].astype(int)

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

    l1_data = data[data["model"] == "L0"]
    sr3_data = data[data["model"] == "SR3_L0"]

    agg_data = sr3_data.copy()
    agg_data = agg_data.merge(l1_data, on="nnz", suffixes=("_sr3", "_l1"))

    base_size = 5
    fig = plt.figure(figsize=(2 * base_size, 2 * base_size))
    grid = plt.GridSpec(nrows=2, ncols=2)

    #     fe_plot = fig.add_subplot(grid[0, 0])
    #     fe_plot.scatter(agg_data["fe_fpr_sr3"], agg_data["fe_sensitivity_sr3"], label="sr3")
    #     fe_plot.scatter(agg_data["fe_fpr_l1"], agg_data["fe_sensitivity_l1"], label="l0")
    #     fe_plot.set_xlabel("FPR FE")
    #     fe_plot.set_ylabel("TPR FE")
    #     fe_plot.legend()

    #     re_plot = fig.add_subplot(grid[0, 1])
    #     re_plot.scatter(agg_data["re_fpr_sr3"], agg_data["re_sensitivity_sr3"], label="sr3")
    #     re_plot.scatter(agg_data["re_fpr_l1"], agg_data["re_sensitivity_l1"], label="l0")
    #     re_plot.set_xlabel("FPR RE")
    #     re_plot.set_ylabel("TPR RE")
    #     re_plot.legend()

    #     all_plot = fig.add_subplot(grid[0, 2])
    #     all_plot.scatter(agg_data["fpr_sr3"], agg_data["sensitivity_sr3"], label="sr3")
    #     all_plot.scatter(agg_data["fpr_l1"], agg_data["sensitivity_l1"], label="l0")
    #     all_plot.set_xlabel("FPR")
    #     all_plot.set_ylabel("TPR")
    #     all_plot.legend()

    fe_plot = fig.add_subplot(grid[0, :])
    # fe_plot.plot(agg_data["nnz"], agg_data["f1_l1"], label="F1 L0")
    fe_plot.plot(agg_data["nnz"], agg_data["fe_f1_l1"], label="L0")
    fe_plot.plot(agg_data["nnz"], agg_data["fe_f1_sr3"], label="L0 SR3")
    fe_plot.legend(loc="upper left")
    fe_plot.set_xlabel(r"$k$, number of allowed NNZ coeffixients")
    fe_plot.set_ylabel(r"F1, selection quality for fixed effects")
    fe_plot.set_title("Fixed-effects selection quality along L0-path")

    re_plot = fig.add_subplot(grid[1, :])
    # re_plot.plot(agg_data["nnz"], agg_data["f1_sr3"], label="F1 SR3")
    re_plot.plot(agg_data["nnz"], agg_data["re_f1_l1"], label="L0")
    re_plot.plot(agg_data["nnz"], agg_data["re_f1_sr3"], label="L0 SR3")
    re_plot.legend(loc="upper left")
    re_plot.set_xlabel(r"$k$, number of allowed NNZ coeffixients")
    re_plot.set_ylabel(r"F1, selection quality for random effects")
    re_plot.set_title("Random-effects selection quality along L0-path")

    #     lambda_l1_plot = fig.add_subplot(grid[3, :])
    #     lambda_l1_plot.plot(agg_data["nnz"], agg_data["tp_l1"], label="tp L0")
    #     lambda_l1_plot.plot(agg_data["nnz"], agg_data["fe_tp_l1"], label="tp FE L0")
    #     lambda_l1_plot.plot(agg_data["nnz"], agg_data["re_tp_l1"], label="tp RE L0")
    #     lambda_l1_plot.legend()

    #     lambda_sr3_plot = fig.add_subplot(grid[4, :])
    #     lambda_sr3_plot.plot(agg_data["nnz"], agg_data["tp_sr3"], label="tp SR3")
    #     lambda_sr3_plot.plot(agg_data["nnz"], agg_data["fe_tp_sr3"], label="tp FE SR3")
    #     lambda_sr3_plot.plot(agg_data["nnz"], agg_data["re_tp_sr3"], label="tp RE SR3")
    #     lambda_sr3_plot.legend()

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    filename = Path(figures_folder) / f"l0_comparison_{suffix if suffix else ''}.pdf"
    plt.savefig(filename)
    print(f"L0 experiment: figure saved as {filename}")

