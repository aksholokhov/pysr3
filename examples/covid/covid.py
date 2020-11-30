from pathlib import Path

import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import normalize

from skmixed.lme.models import LinearLMESparseModel
from skmixed.lme.problems import LinearLMEProblem

from examples.general.settings import thesis_presentation_figures, thesis_proposal_figures,\
    thesis_proposal_tables, presentation_background_color

data_path = Path("/Users/aksh/Storage/repos/IHME_data/covid-19/seir-pipeline-outputs/")
covariates_version = "2020_05_09.01.10"
regression_version = "vis_test_3"
betas_path = data_path / 'regression' / regression_version / "betas"
coefficients_path = data_path / 'regression' / regression_version / "coefficients" / "coefficients_0.csv"
reg_settings_path = data_path / 'regression' / regression_version / "settings.json"
covariates_path = data_path / 'covariate' / covariates_version / "cached_covariates.csv"
cov_metadata_path = data_path / 'covariate' / covariates_version / "metadata.yaml"
location_metadata_path = data_path / 'metadata-inputs' / 'location_metadata_664.csv'

locations_to_get = ("Alaska", "Slovenia", "Turkey", "Switzerland")


def format_xaxis(ax, start_date, end_date, major_tick_interval_days=14, margins_days=5):
    months = mdates.DayLocator(interval=major_tick_interval_days)
    days = mdates.DayLocator()  # Every day
    months_fmt = mdates.DateFormatter('%m/%d')

    # format the ticks
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(days)

    # round to nearest years.
    datemin = np.datetime64(start_date, 'D') - np.timedelta64(margins_days, 'D')
    datemax = np.datetime64(end_date, 'D') + np.timedelta64(margins_days, 'D')
    ax.set_xlim(datemin, datemax)

    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')


def launch_covid_experiment(num_groups=60, presentation = True):
    from matplotlib import rcParams

    rcParams['font.family'] = 'monospace'

    # Read the data
    # num_groups = 15
    coefficients_df = pd.read_csv(coefficients_path)
    groups = list(coefficients_df['group_id'].unique())
    groups = groups[:num_groups]
    betas_df = {group: pd.read_csv(betas_path / f'{group}' / 'regression_draw_0.csv') for group in groups}
    reg_settings = json.load(open(reg_settings_path))
    covariates = list(reg_settings['covariates'].keys())
    covariates.remove("intercept")  # the skmixed model adds its own intercept
    location_metadata = pd.read_csv(location_metadata_path)
    id2loc = location_metadata.set_index('location_id')['location_name'].to_dict()

    # Combine all the data into one matrix
    all_betas = pd.concat(betas_df.values(), axis=0)
    # TODO: Figure out proper std
    all_betas["std"] = 0.1
    all_betas[covariates] = normalize(all_betas[covariates].to_numpy(), axis=0)
    X = all_betas[["location_id"] + covariates + ["std"]].to_numpy()
    columns_labels = [0] + len(covariates) * [3] + [4]
    y = all_betas["beta"].to_numpy()
    problem = LinearLMEProblem.from_x_y(X, y, columns_labels)
    print("done data")

    # Fit the model
    model = LinearLMESparseModel(nnz_tbeta=3, nnz_tgamma=2,
                                 initializer="EM", n_iter_inner=1000, n_iter_outer=1, solver="ip")
    model.fit_problem(problem)
    all_betas["prediction"] = model.predict(X, columns_labels=columns_labels)
    logger = model.logger_
    print("done fitting dense")

    model_sparse = LinearLMESparseModel(nnz_tbeta=3, nnz_tgamma=2,
                                        regularization_type="l2", initializer="EM", solver="ip")
    model_sparse.fit_problem(problem)
    all_betas["sparse_prediction"] = model_sparse.predict(X, columns_labels=columns_labels,
                                                          use_sparse_coefficients=True)
    print("done fitting sparse")

    # Evaluate the model
    def print_on_plot(elements, ax, x0=0.2, y0=9, h=0.4):
        offset = 0
        for s in elements:
            if type(s) is tuple:
                ax.text(x0, y0 - offset, s[0], color=s[1])
                offset += h
            elif type(s) is str:
                ax.text(x0, y0 - offset, s)
                offset += h
            else:
                raise ValueError("wrong type: %s" % type(s))

    ihme_scores = []
    me_scores = []
    groups_description_columns = ["Location", "Obs", "Start", "End"]
    groups_description = pd.DataFrame(columns=groups_description_columns)
    groups_fit_columns = ["Location", "Intercept", "Mobility", "RMSE_IHME", "RMSE_Dense", "RMSE_Sparse"]
    groups_fit = pd.DataFrame(columns=groups_fit_columns)
    for i, group in enumerate(groups):
        fig = plt.figure(figsize=(12, 7))
        grid = plt.GridSpec(1, 2)
        ax = fig.add_subplot(grid[0, 0])
        cur_betas = all_betas[all_betas["location_id"] == group]
        time = pd.to_datetime(cur_betas['date'])
        ax.plot(time, cur_betas["beta"], label='Data')
        ax.plot(time, cur_betas["beta_pred"], label='IHME')
        ax.plot(time, cur_betas["sparse_prediction"], label="R&S Mixed")
        ax.plot(time, cur_betas["prediction"], label="Dense MM")
        #        ax.plot(time, cur_betas["weighted_sparse_prediction"], label="R&S + W Sparse")
        ax.legend()
        start_date = time.to_list()[0]
        end_date = time.to_list()[-1]
        format_xaxis(ax, start_date, end_date)
        ax.set_title(f"{id2loc[group]}")
        ax.set_ylabel(r"$\beta(t)$, contact rate")
        ax.set_xlabel("time, days")
        ax2 = fig.add_subplot(grid[0, 1])
        ax2.set_xlim((0, 12))
        ax2.set_ylim((0, 12))
        table_format = "%-21s%10.2e%10.2e%10.2e%10.2e"
        ihme_error = np.linalg.norm(cur_betas["beta"] - cur_betas["beta_pred"])
        dense_error = np.linalg.norm(cur_betas["beta"] - cur_betas["prediction"])
        sparse_error = np.linalg.norm(cur_betas["beta"] - cur_betas["sparse_prediction"])
        #        weighted_sparse_error = np.linalg.norm(cur_betas["beta"] - cur_betas["weighted_sparse_prediction"])
        groups_description = groups_description.append({
            "Location": id2loc[group],
            "Obs": len(cur_betas["beta"]),
            "Start": start_date,
            "End": end_date,
            "Intercept": model.coef_["random_effects"][i, 0],
            "Mobility": model.coef_["random_effects"][i, 2]
        }, ignore_index=True)

        groups_fit = groups_fit.append({
            "Location": id2loc[group],
            "Intercept": model_sparse.coef_["sparse_per_group_coefficients"][i, 0],
            "Mobility": model_sparse.coef_["sparse_per_group_coefficients"][i, 3],
            "RMSE_IHME": ihme_error,
            "RMSE_Dense": dense_error,
            "RMSE_Sparse": sparse_error
        }, ignore_index=True)

        ihme_scores.append(ihme_error)
        me_scores.append(dense_error)

        effect2coef = [0, 2, 3, 4, 5]

        def coef_to_color(beta, gamma):
            if beta == 0:
                return "red"
            elif gamma == 0:
                return "blue"
            else:
                return "black"

        sorted_errors_idx = np.argsort([ihme_error, dense_error, sparse_error])
        colors_for_errors = ["", "", "", ""]
        for k, color in enumerate(["green", "blue", "black"]):
            colors_for_errors[sorted_errors_idx[k]] = color

        # if ihme_error > weighted_sparse_error:
        #    counter += 1
        diff_dense = dense_error / ihme_error - 1
        diff_sparse = sparse_error / ihme_error - 1
        statistics = [
                         "RMSE:",
                         # "%-12s: %.2e" % ("  IHME", ihme_error),
                         # "%-12s: %.2e" % ("  Dense MM", dense_error),
                         ("%-12s: %.2e" % ("  IHME", ihme_error)),
                         ("%-12s: %.2e %s" % (
                         "  Dense MM", dense_error, f"  {'+' if diff_dense > 0 else ''}{diff_dense:.0%}")),
                         ("%-12s: %.2e %s" % (
                         "  R&S Mixed", sparse_error, f"  {'+' if diff_sparse > 0 else ''}{diff_sparse:.0%}")),
                         # ("%-12s: %.2e" % ("  R&S + W", weighted_sparse_error), colors_for_errors[3]),
                         "",
                     ] + \
                     [
                         "Full MM Coefficients:",
                         "%-21s%10s%10s%10s%10s" % ("name", "local", "mean", "RE", "Var"),
                     ] + \
                     [
                         table_format % (""
                                         "  " + covariate,
                                         model.coef_["per_group_coefficients"][i, effect2coef[j]],
                                         model.coef_["beta"][j],
                                         model.coef_["random_effects"][i, j],
                                         model.coef_["gamma"][j])
                         for j, covariate in enumerate(["intercept"] + covariates)
                     ] + \
                     [
                         "\n",
                         "R&S Mixed Coefficients:",
                         "%-21s%10s%10s%10s%10s" % ("name", "local", "mean", "RE", "Var"),
                     ] + \
                     [
                         (table_format % (""
                                          "  " + covariate,
                                          model_sparse.coef_["sparse_per_group_coefficients"][i, effect2coef[j]],
                                          model_sparse.coef_["tbeta"][j],
                                          model_sparse.coef_["sparse_random_effects"][i, j],
                                          model_sparse.coef_["tgamma"][j]),
                          coef_to_color(model_sparse.coef_["tbeta"][j], model_sparse.coef_["tgamma"][j]))
                         for j, covariate in enumerate(["intercept"] + covariates)
                     ] + \
                     [
                         "\n",
                         "Legend:",
                         "  Both Fixed and Random",
                         ("  Fixed Only", "Blue"),
                         ("  Excluded", "Red")
                     ]

        print_on_plot(statistics, ax2, x0=-1, y0=12)
        ax2.axis('off')
        if id2loc[group] in locations_to_get:
            plt.savefig(thesis_proposal_figures / f"fit_{id2loc[group]}.pdf")
            plt.savefig(thesis_presentation_figures / f"fit_{id2loc[group]}.pdf", facecolor=presentation_background_color)

        plt.close(fig)
    groups_description[groups_description_columns].to_latex(thesis_proposal_tables / "covid_groups_description.tex",
                                                            longtable=True,
                                                            label="table:covid_data_description",
                                                            index=False,
                                                            caption="List of locations, number of observations, "
                                                                    "start and end date for each location "
                                                                    "for COVID-19 Contact Rate Focecasting data")
    groups_fit[groups_fit_columns].to_latex(thesis_proposal_tables / "covid_coefficients.tex",
                                            longtable=True,
                                            label="table:covid_coefficients",
                                            index=False,
                                            float_format="%.2f",
                                            caption=("List of location-specific coefficients" +
                                                     " for the R\\&S-Mixed model fit, " +
                                                     " as well as RMSEs for three models" +
                                                     " discussed in the respective chapter" +
                                                     "Coefficient for \\texttt{temperature} was set to " +
                                                     f"{model_sparse.coef_['sparse_per_group_coefficients'][0, 2]:.2f}. " +
                                                     "Coefficients for \\texttt{proportion\\_over\\_1k}" +
                                                     " and \\texttt{testing\\_reference}" +
                                                     " were set to 0."))
    groups_description[groups_description_columns].to_csv(thesis_proposal_tables / "covid_groups_description.csv")
    groups_fit[groups_fit_columns].to_csv(thesis_proposal_tables / "covid_coefficients.csv")

    # plt.figure(figsize=(8, 8))
    # plt.scatter(ihme_scores, me_scores)
    # plt.xlabel("RMSE IHME")
    # plt.ylabel("RMSE Aleksei")
    # max_err = max(max(ihme_scores), max(me_scores))
    # plt.plot([0, max_err], [0, max_err], '--b')
    # plt.savefig("comparison_0.pdf")


if __name__ == "__main__":
    launch_covid_experiment()

    # pass
