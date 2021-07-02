import numpy as np
import scipy as sp
import pandas as pd
import time
import datetime
import pickle

from pathlib import Path
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import SR3L1LmeModel, L1LmeModel
from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle

from tqdm import tqdm

def run_practical_comparison_experiment(num_trials, num_covariates, model_parameters, problem_parameters, practical_initials, sr3_initials,
                                        logs_folder=".", figures_folder=".", tables_folder="."):
    log = pd.DataFrame(columns=("i", "lambda", "model", "time", "mse", "evar", "loss",
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

            lam = 0.0
            all_coefficients_are_dead = False

            while not all_coefficients_are_dead:
                l1_converged = 1
                l1_SR3_converged = 1

                l1_SR3_practical = SR3L1LmeModel(**model_parameters,
                                      stepping="fixed",
                                      lam=lam,
                                      practical=True,
                                      update_prox_every=5,
                                      )
                l1_SR3_full = SR3L1LmeModel(**model_parameters,
                                             stepping="fixed",
                                             lam=lam,
                                             practical=False)
                tic = time.perf_counter()
                toc = tic
                try:
                    l1_SR3_practical.fit_problem(problem, initial_parameters=practical_initials)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    l1_converged = 0
                finally:
                    l1_y_pred = l1_SR3_practical.predict_problem(problem)

                    l1_results = {
                        "i": i,
                        "lam": lam,
                        "model": "SR3_practical",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, l1_y_pred),
                        "evar": explained_variance_score(y, l1_y_pred),
                        "loss": l1_SR3_practical.logger_.get("loss")[-1],
                        "fe_tp": np.sum((true_beta != 0) & (l1_SR3_practical.coef_["beta"] != 0)),
                        "fe_tn": np.sum((true_beta == 0) & (l1_SR3_practical.coef_["beta"] == 0)),
                        "fe_fp": np.sum((true_beta == 0) & (l1_SR3_practical.coef_["beta"] != 0)),
                        "fe_fn": np.sum((true_beta != 0) & (l1_SR3_practical.coef_["beta"] == 0)),
                        "re_tp": np.sum((true_gamma != 0) & (l1_SR3_practical.coef_["gamma"] != 0)),
                        "re_tn": np.sum((true_gamma == 0) & (l1_SR3_practical.coef_["gamma"] == 0)),
                        "re_fp": np.sum((true_gamma == 0) & (l1_SR3_practical.coef_["gamma"] != 0)),
                        "re_fn": np.sum((true_gamma != 0) & (l1_SR3_practical.coef_["gamma"] == 0)),
                        "number_of_iterations": len(l1_SR3_practical.logger_.get("loss")),
                        "converged": l1_converged
                    }
                    log = log.append(l1_results, ignore_index=True)
                    # l1_initials["beta"] = l1_SR3_practical.coef_["beta"]
                    # l1_initials["gamma"] = l1_SR3_practical.coef_["gamma"]
                tic = time.perf_counter()
                toc = tic
                try:
                    l1_SR3_full.fit_problem(problem, initial_parameters=sr3_initials)
                    toc = time.perf_counter()
                except np.linalg.LinAlgError:
                    toc = time.perf_counter()
                    l1_SR3_converged = 0
                finally:
                    l1_sr3_y_pred = l1_SR3_full.predict_problem(problem)

                    l1_sr3_results = {
                        "i": i,
                        "lam": lam,
                        "model": "SR3_full",
                        "time": toc - tic,
                        "mse": mean_squared_error(y, l1_sr3_y_pred),
                        "evar": explained_variance_score(y, l1_sr3_y_pred),
                        "loss": l1_SR3_full.logger_.get("loss")[-1],
                        "fe_tp": np.sum((true_beta != 0) & (l1_SR3_full.coef_["beta"] != 0)),
                        "fe_tn": np.sum((true_beta == 0) & (l1_SR3_full.coef_["beta"] == 0)),
                        "fe_fp": np.sum((true_beta == 0) & (l1_SR3_full.coef_["beta"] != 0)),
                        "fe_fn": np.sum((true_beta != 0) & (l1_SR3_full.coef_["beta"] == 0)),
                        "re_tp": np.sum((true_gamma != 0) & (l1_SR3_full.coef_["gamma"] != 0)),
                        "re_tn": np.sum((true_gamma == 0) & (l1_SR3_full.coef_["gamma"] == 0)),
                        "re_fp": np.sum((true_gamma == 0) & (l1_SR3_full.coef_["gamma"] != 0)),
                        "re_fn": np.sum((true_gamma != 0) & (l1_SR3_full.coef_["gamma"] == 0)),
                        "number_of_iterations": len(l1_SR3_full.logger_.get("loss")),
                        "converged": l1_SR3_converged
                    }
                    log = log.append(l1_sr3_results, ignore_index=True)
                    # SR3_initials["beta"] = l1_SR3_full.coef_["beta"]
                    # SR3_initials["gamma"] = l1_SR3_full.coef_["gamma"]

                lam = 1.05 * (lam + 0.1)
                print(f"lam={lam}, practical fe = {sum(l1_SR3_practical.coef_['beta'] != 0)}," +
                      f" practical re = {sum(l1_SR3_practical.coef_['gamma'] != 0)}, "+
                      f" full fe = {sum(l1_SR3_full.coef_['beta'] != 0)}," +
                      f" full re = {sum(l1_SR3_full.coef_['gamma'] != 0)}")

                if all(l1_SR3_practical.coef_["beta"] == 0) and all(l1_SR3_practical.coef_["gamma"] == 0) and all(
                        l1_SR3_full.coef_["beta"] == 0) and all(l1_SR3_full.coef_["gamma"] == 0):
                    all_coefficients_are_dead = True

    finally:
        now = datetime.datetime.now()
        log_filename = Path(logs_folder) / f"log_practical_{now}.csv"
        log.to_csv(log_filename)
        print(f"Practical vs Full: data saved as {log_filename}")
        with open(Path(logs_folder) / f"params_practical_vs_full_{now}.csv", 'wb') as f:
            pickle.dump({
                "num_trials": num_trials,
                "num_covariates": num_covariates,
                "model_parameters": model_parameters,
                "problem_parameters": problem_parameters,
                "practical_initials": practical_initials,
                "full_initials": sr3_initials
            }, f)
    return log, now

def plot_practical_comparison(data, suffix=None, figures_folder="."):
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

    practical = data[data["model"] == "SR3_practical"]
    sr3_data = data[data["model"] == "SR3_full"]

    agg_data = sr3_data.copy()
    agg_data = agg_data.merge(practical, on="lam", suffixes=("_sr3", "_practical"))

    base_size = 5
    fig = plt.figure(figsize=(3 * base_size, 2 * base_size))
    grid = plt.GridSpec(nrows=3, ncols=2)

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
    fe_plot.semilogx(agg_data["lam"], agg_data["fe_f1_practical"], label="F1 for FE selection with L1 SR3 Practical")
    fe_plot.semilogx(agg_data["lam"], agg_data["fe_f1_sr3"], label="F1 for FE selection with L1 SR3 Full")
    fe_plot.legend(loc="lower left")
    fe_plot.set_xlabel(r"$\lambda$, strength of LASSO regularizer")
    fe_plot.set_ylabel(r"F1, selection quality for fixed effects")
    fe_plot.set_title("Fixed-effects selection quality along LASSO path")

    re_plot = fig.add_subplot(grid[1, :2])
    # re_plot.semilogx(agg_data["lam"], agg_data["f1_sr3"], label="F1 SR3")
    re_plot.semilogx(agg_data["lam"], agg_data["fe_f1_practical"], label="F1 for RE selection with L1 SR3 Practical")
    re_plot.semilogx(agg_data["lam"], agg_data["re_f1_sr3"], label="F1 for RE selection with L1 SR3 Full")
    re_plot.legend(loc="lower left")
    re_plot.set_xlabel(r"$\lambda$, strength of LASSO regularizer")
    re_plot.set_ylabel(r"F1, selection quality for random effects")
    re_plot.set_title("Random-effects selection quality along LASSO path")

    time_plot = fig.add_subplot(grid[2, :2])
    # re_plot.semilogx(agg_data["lam"], agg_data["f1_sr3"], label="F1 SR3")
    time_plot.loglog(agg_data["lam"], agg_data["time_practical"], label="Time for L1 SR3 Practical")
    time_plot.loglog(agg_data["lam"], agg_data["time_sr3"], label="Time for L1 SR3 Full")
    time_plot.legend(loc="lower left")
    time_plot.set_xlabel(r"$\lambda$, strength of LASSO regularizer")
    time_plot.set_ylabel(r"Time until convergence")
    time_plot.set_title("Time-to-converge along LASSO path")


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
    plot_filename = Path(figures_folder) / f"l1_practical_comparison_{suffix if suffix else ''}.pdf"
    plt.savefig(plot_filename)
    print(f"Practical vs Full: plot saved as {plot_filename}")