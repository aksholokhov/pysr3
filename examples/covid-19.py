from pathlib import Path
from sys import exit

import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import normalize

from skmixed.lme.models import LinearLMESparseModel

data_path = Path("/Users/aksh/Storage/repos/covid-data/covid-19/seir-pipeline-outputs/")
covariates_version = "2020_05_09.01.10"
regression_version = "vis_test_3"
betas_path = data_path / 'regression' / regression_version / "betas"
coefficients_path = data_path / 'regression' / regression_version / "coefficients" / "coefficients_0.csv"
reg_settings_path = data_path / 'regression' / regression_version / "settings.json"
covariates_path = data_path / 'covariate' / covariates_version / "cached_covariates.csv"
cov_metadata_path = data_path / 'covariate' / covariates_version / "metadata.yaml"
location_metadata_path = data_path / 'metadata-inputs' / 'location_metadata_664.csv'


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


if __name__ == "__main__":
    from matplotlib import rcParams

    rcParams['font.family'] = 'monospace'

    # Read the data
    num_groups = 60
    covariates_df = pd.read_csv(covariates_path)
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
    all_betas["std"] = 0.1
    #all_betas[covariates] = normalize(all_betas[covariates].to_numpy(), axis=0)
    X = all_betas[["location_id"] + covariates + ["std"]].to_numpy()
    columns_labels = [0] + len(covariates) * [3] + [4]
    y = all_betas["beta"].to_numpy()
    print("done data");

    # Fit the model
    model = LinearLMESparseModel(lb=0, lg=0, initializer="EM", n_iter=100, n_iter_inner=100)
    model.fit(X, y, columns_labels=columns_labels)
    logger = model.logger_
    all_betas["prediction"] = model.predict(X, columns_labels=columns_labels)
    print("done fitting");


    # Evaluate the model

    def print_on_plot(elements, ax, x0=0.2, y0=9, h=0.3):
        offset = 0
        for s in elements:
            ax.text(x0, y0 - offset, s)
            offset += h


    fig = plt.figure(figsize=(12, 6 * len(groups)))
    grid = plt.GridSpec(len(groups), 2)
    counter = 0
    for i, group in enumerate(groups):
        ax = fig.add_subplot(grid[i, 0])
        cur_betas = all_betas[all_betas["location_id"] == group]
        time = pd.to_datetime(cur_betas['date'])
        ax.plot(time, cur_betas["beta"], label='True')
        ax.plot(time, cur_betas["beta_pred"], label='IHME')
        ax.plot(time, cur_betas["prediction"], label="Aleksei")
        ax.legend()
        start_date = time.to_list()[0]
        end_date = time.to_list()[-1]
        format_xaxis(ax, start_date, end_date)
        ax.set_title(f"{regression_version}: {id2loc[group]}")
        ax2 = fig.add_subplot(grid[i, 1])
        ax2.set_xlim((0, 10))
        ax2.set_ylim((0, 10))
        table_format = "%-25s%10.4f%10.4f%10.4f%10.4f"
        ihme_error = np.linalg.norm(cur_betas["beta"] - cur_betas["beta_pred"])
        my_error = np.linalg.norm(cur_betas["beta"] - cur_betas["prediction"])
        if ihme_error > my_error:
            counter += 1
        statistics = [
                         "RMSE:",
                         "%6s: %.3f" %("IHME", ihme_error),
                         "%6s: %.3f" %("Me", my_error),
                         "",
                         "Coefficients:",
                         "%-25s%10s%10s%10s%10s" % ("name", "local", "mean", "RE", "Var")
                     ] + \
                     [
                         table_format % (covariate,
                                         model.coef_["per_group_coefficients"][i, j],
                                         model.coef_["beta"][j],
                                         model.coef_["random_effects"][i, j],
                                         model.coef_["gamma"][j])
                         for j, covariate in enumerate(["intercept"] + covariates)
                     ]
        print_on_plot(statistics, ax2)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

    plt.show()
    print(counter/len(groups))



