from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.models import LinearLMESparseModel

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex

from examples.general.settings import thesis_presentation_figures

from examples.general.settings import presentation_background_color as background_color

if __name__ == "__main__":


    # plt.rcParams['figure.facecolor'] = background_color

    trials = 20
    problem_parameters = {
        "groups_sizes": [20, 5, 10, 50],
        "features_labels": [3],
        "random_intercept": True,
        "seed": 41,
        "gamma": np.array([0.7, 0.6]),
        "obs_std": 0.3
    }
    model_parameters = {
        "nnz_tbeta": 2,
        "nnz_tgamma": 2,
        "lb": 0,        # We expect the coefficient vectors to be dense so we turn regularization off.
        "lg": 0,        # Same.
        "initializer": 'EM',
        "solver": 'pgd',
        "logger_keys": ('converged', 'loss',),
        "tol_inner": 1e-6,
        "tol_outer": 1e-6,
        "n_iter_inner": 1000,
        "n_iter_outer": 1  # we don't care about tbeta and tgamma, so we don't increase regularization iteratively
    }

    problem, true_model_parameters = LinearLMEProblem.generate(**problem_parameters)
    model = LinearLMESparseModel(**model_parameters)
    model.fit_problem(problem)
    beta = model.coef_['beta']
    per_group_coefs = model.coef_["per_group_coefficients"]

    colors = sns.color_palette("Set2", problem.num_groups)

    # just data
    plt.figure(figsize=(6, 6))
    x_bounds = (-2, 2)
    for i, (x, y, z, l) in enumerate(problem):
        plt.scatter(x[:, 1], y, label=f"Group {i+1}", color=to_hex(colors[i]))
    # plt.plot(x_bounds, [beta[0] + x_bounds[0]*beta[1], beta[0] + x_bounds[1]*beta[1]], label="Mean prediction")
    plt.legend()
    plt.xlim((-3, 3))
    plt.ylim((-1, 4))
    plt.xlabel("X, feature")
    plt.ylabel("Y, target")
    plt.savefig(thesis_presentation_figures / "lme_example_data_only.pdf", facecolor=background_color)

    # data with mean prediction
    plt.figure(figsize=(6, 6))
    x_bounds = (-2, 2)
    for i, (x, y, z, l) in enumerate(problem):
        plt.scatter(x[:, 1], y, label=f"Group {i+1}", color=to_hex(colors[i]))
    plt.plot(x_bounds, [beta[0] + x_bounds[0]*beta[1], beta[0] + x_bounds[1]*beta[1]], label="Mean prediction")
    plt.legend()
    plt.xlim((-3, 3))
    plt.ylim((-1, 4))
    plt.xlabel("X, feature")
    plt.ylabel("Y, target")
    plt.savefig(thesis_presentation_figures / "lme_example_mean_prediction.pdf", facecolor=background_color)
    plt.close()

    # data with mixed model prediction
    plt.figure(figsize=(6, 6))
    x_bounds = (-2, 2)
    for i, ((x, y, z, l), coefs) in enumerate(zip(problem, per_group_coefs)):
        plt.scatter(x[:, 1], y, label=f"Group {i+1}", color=to_hex(colors[i]))
        plt.plot(x_bounds, [coefs[0] + x_bounds[0] * coefs[3], beta[0] + x_bounds[1] * coefs[3]], color=to_hex(colors[i]))
    plt.legend()
    plt.xlim((-3, 3))
    plt.ylim((-1, 4))
    plt.xlabel("X, feature")
    plt.ylabel("Y, target")
    plt.savefig(thesis_presentation_figures / "lme_example_random_prediction.pdf", facecolor=background_color)
    plt.close()
    pass
