from pathlib import Path

import numpy as np
import scipy as sp

from examples.paper_sum2021.intuition import run_intuition_experiment, plot_intuition_picture
from examples.paper_sum2021.l0_experiment import run_l0_comparison_experiment, plot_l0_comparison
from examples.paper_sum2021.l1_experiment import run_l1_comparison_experiment, plot_l1_comparison
from examples.paper_sum2021.practical_vs_full import run_practical_comparison_experiment, plot_practical_comparison

base_folder = Path("/Users/aksh/Storage/repos/skmixed/examples/paper_sum2021")

logs_folder = base_folder / "logs"
figures_folder = base_folder / "figures"
tables_folder = base_folder / "tables"

experiments_to_launch = {
    "intuition": False,
    "practical_vs_full_l1": False,
    "l0_comparison": False,
    "l1_comparison": True,
    "CAD_comparison": False,
    "krishna_setup_comparison": False
}

release = False

if __name__ == "__main__":
    if experiments_to_launch.get("intuition", False):
        print("Run intuition experiment")
        seed = 13
        num_covariates = 2

        correlation_between_adjacent_covariates = 0.0
        beta_lims = (-3, 3)
        gamma_lims = (0, 3)
        grid_dim = 100

        results, now = run_intuition_experiment(
            seed=seed,
            num_covariates=num_covariates,
            model_parameters={
                "lb": 5,
                "lg": 5,
            },
            problem_parameters={
                "groups_sizes": [10, 15, 4, 8, 3, 5],
                "features_labels": [3] * num_covariates,
                "random_intercept": False,
                "obs_std": 0.1,
                "chance_missing": 0,
                "chance_outlier": 0.0,
                "outlier_multiplier": 5,
                # "features_covariance_matrix": np.eye(num_covariates)
                "features_covariance_matrix": sp.linalg.block_diag(
                    *([np.array([[1, correlation_between_adjacent_covariates],
                                 [correlation_between_adjacent_covariates, 1]])] * int(num_covariates / 2)))
            },
            lam=1.5,
            initial_parameters={
                "beta": -np.ones(3),
                "gamma": np.array([1, 2])
            },
            beta_lims=beta_lims,
            gamma_lims=gamma_lims,
            grid_dim=grid_dim,
            logs_folder=logs_folder
        )
        plot_intuition_picture(results, suffix="" if release else now,
                               beta_lims=beta_lims,
                               gamma_lims=gamma_lims,
                               grid_dim=grid_dim,
                               figures_folder=figures_folder)

    if experiments_to_launch.get("l0_comparison", False):
        print("Run L0 comparison experiment")
        num_covariates = 40
        correlation_between_adjacent_covariates = 0.0
        logs_l0, now = run_l0_comparison_experiment(
            num_trials=1,
            num_covariates=num_covariates,
            model_parameters={
                "lb": 40,
                "lg": 40,
                "initializer": "None",
                "logger_keys": ('converged', 'loss',),
                "tol_oracle": 1e-3,
                "tol_solver": 1e-6,
                "max_iter_oracle": 10,
                "max_iter_solver": 10000,
            },
            problem_parameters = {
                "groups_sizes": [20, 12, 14, 50, 11] * 2,
                "features_labels": [3] * num_covariates,
                "random_intercept": True,
                "obs_std": 0.1,
                "chance_missing": 0,
                "chance_outlier": 0.0,
                "outlier_multiplier": 5,
                # "features_covariance_matrix": np.eye(num_covariates)
                "features_covariance_matrix": sp.linalg.block_diag(
                    *([np.array([[1, correlation_between_adjacent_covariates],
                                 [correlation_between_adjacent_covariates, 1]])] * int(num_covariates / 2)))
            },
            l0_initials={
                "beta": np.ones(num_covariates + 1),
                "gamma": np.ones(num_covariates + 1)
            },
            sr3_initials={
                "beta": np.ones(num_covariates + 1),
                "gamma": np.ones(num_covariates + 1)
            },
            logs_folder=logs_folder
        )
        plot_l0_comparison(logs_l0, suffix="" if release else now, figures_folder=figures_folder)

    if experiments_to_launch.get("l1_comparison", False):
        print("Run L1 comparison experiment")
        num_covariates = 20
        correlation_between_adjacent_covariates = 0.0
        logs_l1, now = run_l1_comparison_experiment(
            num_covariates=num_covariates,
            num_trials = 1,
            model_parameters = {
                "lb": 40,
                "lg": 40,
                "initializer": "None",
                "logger_keys": ('converged', 'loss',),
                "tol_oracle": 1e-3,
                "tol_solver": 1e-6,
                "max_iter_oracle": 10000,
                "max_iter_solver": 10000
            },
            problem_parameters = {
                "groups_sizes": [20, 12, 14, 50, 11]*2,
                "features_labels": [3]*num_covariates,
                "random_intercept": True,
                "obs_std": 0.1,
                "chance_missing": 0,
                "chance_outlier": 0.0,
                "outlier_multiplier": 5,
                # "features_covariance_matrix": np.eye(num_covariates)
                "features_covariance_matrix": sp.linalg.block_diag(*([np.array([[1, correlation_between_adjacent_covariates],
                                                                                [correlation_between_adjacent_covariates, 1]])]*int(num_covariates/2)))
            },
            l1_initials = {
                "beta": np.ones(num_covariates + 1),
                "gamma": np.ones(num_covariates + 1)
            },
            sr3_initials = {
                "beta": np.ones(num_covariates + 1),
                "gamma": np.ones(num_covariates + 1)
            },
            logs_folder=logs_folder
        )
        plot_l1_comparison(logs_l1, suffix="" if release else now, figures_folder=figures_folder)

    if experiments_to_launch.get("practical_vs_full_l1", False):
        print("Run practical vs full comparison experiment")
        num_covariates = 20
        correlation_between_adjacent_covariates = 0.0
        logs_practical, now = run_practical_comparison_experiment(
            num_covariates=num_covariates,
            num_trials=1,
            model_parameters={
                "lb": 40,
                "lg": 40,
                "initializer": "None",
                "logger_keys": ('converged', 'loss',),
                "tol_oracle": 1e-3,
                "tol_solver": 1e-6,
                "max_iter_oracle": 10000,
                "max_iter_solver": 10000
            },
            problem_parameters={
                "groups_sizes": [20, 12, 14, 50, 11] * 2,
                "features_labels": [3] * num_covariates,
                "random_intercept": True,
                "obs_std": 0.1,
                "chance_missing": 0,
                "chance_outlier": 0.0,
                "outlier_multiplier": 5,
                # "features_covariance_matrix": np.eye(num_covariates)
                "features_covariance_matrix": sp.linalg.block_diag(
                    *([np.array([[1, correlation_between_adjacent_covariates],
                                 [correlation_between_adjacent_covariates, 1]])] * int(num_covariates / 2)))
            },
            practical_initials={
                "beta": np.ones(num_covariates + 1),
                "gamma": np.ones(num_covariates + 1)
            },
            sr3_initials={
                "beta": np.ones(num_covariates + 1),
                "gamma": np.ones(num_covariates + 1)
            },
            logs_folder=logs_folder
        )
        plot_practical_comparison(logs_practical, suffix="" if release else now, figures_folder=figures_folder)
