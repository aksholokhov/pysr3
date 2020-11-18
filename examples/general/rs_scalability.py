import unittest
import time
from datetime import datetime
import pickle

from pathlib import Path

import numpy as np
from scipy.linalg import block_diag
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score
from matplotlib import pyplot as plt

import pandas as pd

from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.models import LinearLMESparseModel

thesis_repo_path = Path("/Users/aksh/Storage/repos/general_phd_proposal")

config = {
    "trials_per_p": 200,
    "fraction_of_active_fixed_features": 0.5,
    "fraction_of_active_random_features": 0.8,  # out of those active fixed features
    "fixed_groups_sizes": [10] * 6,
    "trajectories_transparency": 0.05
}

problem_parameters = {
    # "groups_sizes": [20, 12, 14, 50, 11],
    # "features_labels": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    "random_intercept": True,
    "obs_std": 0.1,
}

rs_model_parameters = {
    "lb": 0.01,
    "lg": 0.01,
    "initializer": "None",
    "logger_keys": ('converged', 'loss',),
    "tol_inner": 1e-5,
    "n_iter_inner": 5000,
    "tol_outer": 1e-4,
    "n_iter_outer": 50,
}


def generate_data(ps):
    df_columns = ["m", "t", "TIME", "RE_ACC", "FE_ACC", "MSE", "VAR", "MSE_TEST", "VAR_TEST", "ITER"]
    records_rs = pd.DataFrame(columns=df_columns)

    problems = {}

    # ps = [4, 7, 10]
    # ps = range(config["m_start"], config["m_end"], config["m_step"])

    rs_prev_model = 0
    rs_prev_time = 0
    try:
        for i, p in enumerate(ps):
            for t in range(config["trials_per_p"]):
                np.random.seed(42 + 100 * p + t)
                true_beta_idxs = np.random.choice(p, size=int(p * config["fraction_of_active_fixed_features"]),
                                                  replace=False)
                true_beta = np.zeros(p)
                true_beta *= np.random.randn(len(true_beta))
                true_beta[true_beta_idxs] = 1
                true_gamma_idx = np.random.choice(true_beta_idxs, size=int(
                    len(true_beta_idxs) * config["fraction_of_active_random_features"]))
                true_gamma = np.zeros(p)
                true_gamma[true_gamma_idx] = 1
                true_gamma *= abs(np.random.randn(len(true_gamma)))

                nnz_tbeta = int(sum(true_beta != 0))
                nnz_tgamma = int(sum(true_gamma != 0))

                # groups_sizes = np.random.randint(config["group_size_min"], config["group_size_max"]+1, 5)

                problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters,
                                                                           groups_sizes=config["fixed_groups_sizes"],
                                                                           features_labels=[3] * (p - 1),
                                                                           features_covariance_matrix=block_diag(
                                                                               *[np.array([
                                                                                   [9, 4.8, 0.6],
                                                                                   [4.8, 4, 1],
                                                                                   [0.6, 1, 1]
                                                                               ])] * ((p - 1) // 3)),
                                                                           beta=true_beta,
                                                                           gamma=true_gamma,
                                                                           seed=42 + 1000 * p + t)
                test_problem, _ = LinearLMEProblem.generate(**problem_parameters,
                                                            groups_sizes=config["fixed_groups_sizes"],
                                                            features_labels=[3] * (p - 1),
                                                            features_covariance_matrix=block_diag(
                                                                *[np.array([
                                                                    [9, 4.8, 0.6],
                                                                    [4.8, 4, 1],
                                                                    [0.6, 1, 1]
                                                                ])] * ((p - 1) // 3)),
                                                            beta=true_beta,
                                                            gamma=true_gamma,
                                                            true_random_effects=true_model_parameters["random_effects"],
                                                            seed=(42 + 1000 * p + t) * 10)
                x, y = problem.to_x_y()
                x_test, y_test = test_problem.to_x_y()

                rs_model = LinearLMESparseModel(**rs_model_parameters,
                                                nnz_tbeta=nnz_tbeta,
                                                nnz_tgamma=nnz_tgamma)
                rs_model.tol_inner *= p
                if rs_prev_time < 3600:
                    t0 = time.time()
                    rs_model.fit_problem(problem)
                    rs_time = time.time() - t0
                    print(f" {p}-{t}-rs ", end="\n")
                    rs_prev_model = rs_model
                    rs_prev_time = rs_time
                else:
                    rs_model = rs_prev_model
                    rs_time = -1
                    print(f" {p}-{t}-rs-NO ", end="\n")

                problems[(p, t)] = (problem, rs_model)

                # lasso fail checks
                # logger = lasso_model.logger_
                # loss = np.array(logger.get("loss"))
                # assert np.all(loss[1:] - loss[:-1] <= 0), "%d) Loss does not decrease monotonically with iterations. (seed=%d)" % (m, m)

                rs_iterations = len(rs_model.logger_.get("loss"))

                # oracle = LinearLMEOracle(problem)

                rs_y_pred = rs_model.predict_problem(problem, use_sparse_coefficients=True)
                rs_y_pred_test = rs_model.predict_problem(test_problem, use_sparse_coefficients=True)
                rs_explained_variance = explained_variance_score(y, rs_y_pred)
                rs_explained_variance_test = explained_variance_score(y_test, rs_y_pred_test)
                rs_mse = mean_squared_error(y, rs_y_pred)
                rs_mse_test = mean_squared_error(y_test, rs_y_pred_test)

                rs_maybe_tbeta = rs_model.coef_["tbeta"]
                rs_maybe_tgamma = rs_model.coef_["tgamma"]
                # rs_aic = 0 if p > problem.num_obs else oracle.vaida2005aic(rs_maybe_tbeta, rs_maybe_tgamma)
                # rs_bic = oracle.jones2010bic(rs_maybe_tbeta, rs_maybe_tgamma)
                rs_fixed_effects_accuracy = accuracy_score(true_beta != 0, rs_maybe_tbeta != 0)
                rs_random_effects_accuracy = accuracy_score(true_gamma != 0, rs_maybe_tgamma != 0)

                records_rs = records_rs.append({
                    "m": p,
                    "t": t,
                    # "AIC": rs_aic,
                    # "BIC": rs_bic,
                    "TIME": rs_time,
                    "ITER": rs_iterations,
                    "MSE": rs_mse,
                    "VAR": rs_explained_variance,
                    "MSE_TEST": rs_mse_test,
                    "VAR_TEST": rs_explained_variance_test,
                    "FE_ACC": rs_fixed_effects_accuracy,
                    "RE_ACC": rs_random_effects_accuracy
                }, ignore_index=True)
    except Exception as e:
        raise e
    finally:
        now = datetime.now()
        records_rs.to_csv(open("rs_%s.csv" % (now.strftime("%d_%m_%Y_%H_%M_%S")), "w"))
        pickle.dump(problems, open("models_%s.dump" % (now.strftime("%d_%m_%Y_%H_%M_%S")), "wb"))
    return records_rs


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    regenerate_data = False
    trajectories = False
    percentiles = True
    vertical_lines = True
    presentation = True
    ps = [4, 7, 10, 16, 19, 25, 31, 37, 40, 46, 55, 61, 67, 76, 91]#, 106, 121]
    # ps = [22]
    if regenerate_data:
        rs_data = generate_data(ps=ps)
    else:
        rs_data = pd.read_csv("rs_11_11_2020_07_42_06.csv")
        rs_data = rs_data[rs_data["m"].isin(ps)]
    rs_data_mean = rs_data.groupby("m").agg(
        func='median'
    )
    rs_data_low = rs_data.groupby("m").agg(
        func=lambda x: np.percentile(x, 5)
    )
    rs_data_high = rs_data.groupby("m").agg(
        func=lambda x: np.percentile(x, 95)
    )
    #ps = np.array(rs_data_mean.index, dtype=int)
    if presentation:
        fig1 = plt.figure(figsize=(12, 12))
        grid1 = plt.GridSpec(nrows=2, ncols=2)
        fig2 = plt.figure(figsize=(12, 12))
        grid2 = plt.GridSpec(nrows=2, ncols=2)
        time_plot = fig1.add_subplot(grid1[0, 0])
        iterations_plot = fig1.add_subplot(grid1[0, 1])
        time_plot_2 = fig2.add_subplot(grid2[0, 0])
        iterations_plot_2 = fig2.add_subplot(grid2[0, 1])
        fe_re_acc_plot = fig1.add_subplot(grid1[1, :])
        mse_plot = fig2.add_subplot(grid2[1, :])
    else:
        fig = plt.figure(figsize=(12, 18))
        grid = plt.GridSpec(nrows=3, ncols=2)
        time_plot = fig.add_subplot(grid[0, 0])
        iterations_plot = fig.add_subplot(grid[0, 1])
        fe_re_acc_plot = fig.add_subplot(grid[1, :])
        mse_plot = fig.add_subplot(grid[2, :])

    time_plot.semilogy(ps, rs_data_mean["TIME"])
    time_plot.set_ylabel("Time, seconds")
    time_plot.set_title("Time")
    if presentation:
        time_plot_2.semilogy(ps, rs_data_mean["TIME"])
        time_plot_2.set_ylabel("Time, seconds")
        time_plot_2.set_title("Time")

    iterations_plot.semilogy(ps, rs_data_mean["ITER"])
    iterations_plot.set_ylabel("Iterations")
    iterations_plot.set_title("Number of Iterations")
    if presentation:
        iterations_plot_2.semilogy(ps, rs_data_mean["ITER"])
        iterations_plot_2.set_ylabel("Iterations")
        iterations_plot_2.set_title("Number of Iterations")

    fe_re_acc_plot.plot(ps, rs_data_mean["FE_ACC"], label="Fixed Effects", c='b')
    fe_re_acc_plot.plot(ps, rs_data_mean["RE_ACC"], label="Random Effects", c='m')
    fe_re_acc_plot.set_ylabel("Accuracy")
    fe_re_acc_plot.legend()
    fe_re_acc_plot.set_title("Selection Accuracy for Fixed Effects and Random Effects")
    if presentation:
        fe_re_acc_plot.set_xlabel(r"$p$, Number of Covariates in Dataset")

    mse_plot.semilogy(ps, rs_data_mean["MSE"], label="Train", c='b')
    mse_plot.semilogy(ps, rs_data_mean["MSE_TEST"], label="Test", c='m')
    mse_plot.set_ylabel("Mean Squared Error")
    mse_plot.set_xlabel(r"$p$, Number of Covariates in Dataset")
    mse_plot.legend()
    mse_plot.set_title("Mean Squared Error")
    # var_plot = fig.add_subplot(grid[2, 1])
    # var_plot.semilogy(ps, rs_data_mean["VAR"], label="Train", c='b')
    # var_plot.semilogy(ps, rs_data_mean["VAR_TEST"], label="Test", c='r')
    # var_plot.set_ylabel("Explained Variance")
    # var_plot.set_xlabel(r"$p$, Number of Covariates in Dataset")
    # var_plot.legend()
    # var_plot.set_title("Explained Variance")

    # sample trajectories
    if trajectories:
        for t in range(config["trials_per_p"]):
            rs_data_draw = rs_data[rs_data["t"] == t]
            alpha = config["trajectories_transparency"]
            time_plot.semilogy(ps, rs_data_draw['TIME'], c='b', alpha=alpha)
            iterations_plot.semilogy(ps, rs_data_draw["ITER"], c='b', alpha=alpha)
            fe_re_acc_plot.plot(ps, rs_data_draw["FE_ACC"], c='b', alpha=alpha)
            fe_re_acc_plot.plot(ps, rs_data_draw["RE_ACC"], c='m', alpha=alpha)
            mse_plot.semilogy(ps, rs_data_draw["MSE"], c='b', alpha=alpha)
            mse_plot.semilogy(ps, rs_data_draw["MSE_TEST"], c='m', alpha=alpha)
            # var_plot.semilogy(ps, rs_data_draw["VAR"], c='b', alpha=alpha)
            # var_plot.semilogy(ps, rs_data_draw["VAR_TEST"], c='r', alpha=alpha)
    if percentiles:
        alpha = config["trajectories_transparency"]
        time_plot.fill_between(ps, rs_data_low['TIME'], rs_data_high['TIME'], facecolor='b', alpha=alpha)
        iterations_plot.fill_between(ps, rs_data_low["ITER"], rs_data_high["ITER"], facecolor='b', alpha=alpha)
        if presentation:
            time_plot_2.fill_between(ps, rs_data_low['TIME'], rs_data_high['TIME'], facecolor='b', alpha=alpha)
            iterations_plot_2.fill_between(ps, rs_data_low["ITER"], rs_data_high["ITER"], facecolor='b', alpha=alpha)
        fe_re_acc_plot.fill_between(ps, rs_data_low["FE_ACC"], rs_data_high["FE_ACC"], label="95\% Interval", facecolor='b', alpha=alpha)
        fe_re_acc_plot.fill_between(ps, rs_data_low["RE_ACC"], rs_data_high["RE_ACC"], label="95\% Interval", facecolor='m', alpha=alpha)
        mse_plot.fill_between(ps, rs_data_low["MSE"], rs_data_high["MSE"], facecolor='b', alpha=alpha)
        mse_plot.fill_between(ps, rs_data_low["MSE_TEST"], rs_data_high["MSE_TEST"], facecolor='m', alpha=alpha)
        # var_plot.fill_between(ps, rs_data_low["VAR"], rs_data_high["VAR"], facecolor='b', alpha=alpha)
        # var_plot.fill_between(ps, rs_data_low["VAR_TEST"], rs_data_high["VAR_TEST"], facecolor='r', alpha=alpha)

    if vertical_lines:
        first_line = 10
        second_line = 60
        ylim = time_plot.get_ylim()
        time_plot.plot([first_line]*2, ylim, '--', c='black')
        time_plot.plot([second_line]*2, ylim, '--', c='black')
        time_plot.text(first_line / 2, ylim[0], "1", c='orange')
        time_plot.text(first_line + (second_line - first_line)/2, ylim[0], "2", c='orange')
        time_plot.text(second_line + (time_plot.get_xlim()[1] - second_line) / 2, ylim[0], "3", c='orange')
        if presentation:
            time_plot_2.plot([first_line] * 2, ylim, '--', c='black')
            time_plot_2.plot([second_line] * 2, ylim, '--', c='black')
            time_plot_2.text(first_line / 2, ylim[0], "1", c='orange')
            time_plot_2.text(first_line + (second_line - first_line) / 2, ylim[0], "2", c='orange')
            time_plot_2.text(second_line + (time_plot.get_xlim()[1] - second_line) / 2, ylim[0], "3", c='orange')

        ylim = iterations_plot.get_ylim()
        iterations_plot.plot([first_line]*2, ylim, '--', c='black')
        iterations_plot.plot([second_line]*2, ylim, '--', c='black')
        iterations_plot.text(first_line / 2, ylim[0], "1", c='orange')
        iterations_plot.text(first_line + (second_line - first_line) / 2, ylim[0], "2", c='orange')
        iterations_plot.text(second_line + (iterations_plot.get_xlim()[1] - second_line) / 2, ylim[0], "3", c='orange')
        if presentation:
            iterations_plot_2.plot([first_line] * 2, ylim, '--', c='black')
            iterations_plot_2.plot([second_line] * 2, ylim, '--', c='black')
            iterations_plot_2.text(first_line / 2, ylim[0], "1", c='orange')
            iterations_plot_2.text(first_line + (second_line - first_line) / 2, ylim[0], "2", c='orange')
            iterations_plot_2.text(second_line + (iterations_plot.get_xlim()[1] - second_line) / 2, ylim[0], "3",
                                 c='orange')

        ylim = fe_re_acc_plot.get_ylim()
        fe_re_acc_plot.plot([first_line]*2, ylim, '--', c='black')
        fe_re_acc_plot.plot([second_line]*2, ylim, '--', c='black')
        fe_re_acc_plot.text(first_line / 2, ylim[0], "1", c='orange')
        fe_re_acc_plot.text(first_line + (second_line - first_line) / 2, ylim[0], "2", c='orange')
        fe_re_acc_plot.text(second_line + (fe_re_acc_plot.get_xlim()[1] - second_line) / 2, ylim[0], "3", c='orange')
        ylim = mse_plot.get_ylim()
        mse_plot.plot([first_line]*2, ylim, '--', c='black')
        mse_plot.semilogy([second_line]*2, ylim, '--', c='black')
        mse_plot.text(first_line / 2, ylim[0], "1", c='orange')
        mse_plot.text(first_line + (second_line - first_line) / 2, ylim[0], "2", c='orange')
        mse_plot.text(second_line + (mse_plot.get_xlim()[1] - second_line) / 2, ylim[0], "3", c='orange')
    if presentation:
        fig1.savefig(thesis_repo_path / "presentation" / "Figures" / "scalability_accuracy.pdf")
        fig2.savefig(thesis_repo_path / "presentation" / "Figures" / "scalability_mse.pdf")
    else:
        plt.savefig(thesis_repo_path / "Images" / "scalability_experiments.pdf")
    print("done")
