from pathlib import Path

from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score
import pandas as pd
import numpy as np

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import seaborn as sns

from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle, LinearLMEOracleRegularized, LinearLMEOracleW
from skmixed.lme.models import LinearLMESparseModel
from skmixed.lme.trees import Tree, Forest

# %%
figures_folder_path = Path("figures/evidence_score")
datasets_folder_path = Path("/Users/aksh/Storage/repos/skmixed/datasets/evidence_score_data/")
redmeat_datasets = [f for f in datasets_folder_path.iterdir() if str(f.name).startswith("redmeat")]

# %% settings for redmeat_colorectal.csv dataset
# dataset_path = Path("/Users/aksh/Storage/repos/skmixed/datasets/evidence_score_data/redmeat_colorectal.csv")

main_features_columns = ["nid", "se"]

target_column = ["target"]

empty_columns = ["exposure_1",
                 'exposure_2',
                 'outcome_1',
                 'outcome_2',
                 'confounder_1',
                 'beef',
                 'pork',
                 'nonstandard_exp',
                 'selection_bias',
                 'reverse_causation'
                 ]

maybe_continuous_features = [
    "seq",
    'follow_up',
]

categorical_features_columns = [
    'sensitivity',
    'representative',
    'total_red',
    "confounder_2",
    "exposure_3",
    'odds_ratio',
    'mortality',
]

categorical_features_columns = []
# %% generate intercept only solutions
for dataset_path in redmeat_datasets:
    data = pd.read_csv(dataset_path)
    # This is Peng's formula for getting a covariate variable (exposure)
    data["linear_exposure"] = (data["b_0"] + data["b_1"]) / 2 - (data["a_0"] + data["a_1"]) / 2
    # TODO: ask Peng what I should use as SE given the transformation of target
    data["target"] = data["ln_effect"] / data["linear_exposure"]
    # We also normalize data
    target_se = np.sqrt(data["target"].var())
    data["target"] = data["target"] / target_se
    data["se"] = (data["ln_se"] / (np.abs(data["linear_exposure"]) * target_se))
    # data["se"] = 0.1 / np.abs(data["linear_exposure"])
    # Data prep for the model
    groups_to_get = data['nid'].unique().tolist()  # [:4]
    data_short = data[main_features_columns + target_column + categorical_features_columns]
    data_short = data_short[data_short["nid"].isin(groups_to_get)]

    X = data_short[main_features_columns + categorical_features_columns].to_numpy()
    y = data_short[target_column].to_numpy().flatten()
    column_labels = [0, 4] + [3] * len(categorical_features_columns)
    X = np.vstack([column_labels, X])

    # Fitting the model
    problem = LinearLMEProblem.from_x_y(X, y, random_intercept=True, add_group_as_categorical_feature=True)
    model = LinearLMESparseModel(lb=0, lg=0, nnz_tbeta=1, nnz_tgamma=1, n_iter_outer=1, initializer=None, tol=1e-5)
    model.fit_problem(problem)
    y_pred = model.predict_problem(problem)

    # plot solution in the original space
    figure = plt.figure(figsize=(12, 12))
    grid = plt.GridSpec(nrows=2, ncols=2)
    # plot solutions
    solution_plot = figure.add_subplot(grid[0, 0])
    colors = sns.color_palette("Set2", problem.num_groups)
    mean_coefs = []
    for i, (coef, (x, y, z, l)) in enumerate(zip(model.coef_["per_group_coefficients"], problem)):
        group_id = problem.group_labels[i]
        group_data = data[data["nid"] == group_id]
        # transform target variable back to ln_effect space
        exposure = group_data["linear_exposure"]
        ln_effect = y * target_se * exposure
        # coef[0] is intercept coef, coef[1:3] are 0s (group and std), coef[3:] are coefs for categorical covariates
        group_coefficient = (coef[0] + x[0, 1:].dot(coef[3:])) * target_se
        color = to_hex(colors[i])
        solution_plot.scatter(exposure, ln_effect, label=group_id, c=color)
        solution_plot.errorbar(exposure, ln_effect, yerr=group_data['ln_se'], c=colors[i], fmt="none")
        solution_plot.plot([0, max(exposure)], [0, group_coefficient * max(exposure)], c=color)
        mean_coef = np.mean(y)
        # solution_plot.plot([0, max(exposure)], [0, mean_coef*max(exposure)],'--', c=color)
        mean_coefs.append(mean_coef)
    solution_plot.set_title(f"{dataset_path.name}; solution, gamma = {model.coef_['gamma']}")

    # plot residues
    residuals_plot = figure.add_subplot(grid[0, 1])
    data_short["predictions"] = model.predict_problem(problem)
    data_short["residuals"] = (data_short["target"] - data_short["predictions"]) * target_se
    sns.swarmplot(x="nid", y="residuals", data=data_short, ax=residuals_plot, palette=colors)
    residuals_plot.set_xticklabels(range(problem.num_groups))
    residuals_plot.set_title("residuals")

    # plot loss functions and information criteria
    oracle = LinearLMEOracle(problem)
    demarginalized_loss_plot = figure.add_subplot(grid[1, 1])
    loss_plot = figure.add_subplot(grid[1, 0])
    # aic_plot = figure.add_subplot(grid[1, 1])
    demarginalized_losses = []
    losses = []
    aics = []
    gammas = np.linspace(0, 0.1, 100)

    for g in gammas:
        current_gamma = np.array([g])
        current_beta = oracle.optimal_beta(current_gamma)
        losses.append(oracle.loss(current_beta, current_gamma))
        aics.append(oracle.vaida2005aic(current_beta, current_gamma))
        demarginalized_losses.append(oracle.demarginalized_loss(current_beta, current_gamma))
    loss_plot.plot(gammas, losses)
    demarginalized_loss_plot.plot(gammas, demarginalized_losses)
    # aic_plot.plot(gammas, aics)
    loss_plot.set_title("Loss")
    demarginalized_loss_plot.set_title("Demarginalized loss")
    # aic_plot.set_title("AIC")

    plt.savefig(figures_folder_path / f"{dataset_path.name}_intercept_only.png")

# %% plot feature selection plots
print("feature selection starts")
maybe_categorical_features_columns = [
    'sensitivity',
    'representative',
    'total_red',
    "confounder_2",
    "exposure_3",
    'odds_ratio',
    'mortality',
    "exposure_1",
    'exposure_2',
    'outcome_1',
    'outcome_2',
    'confounder_1',
    'beef',
    'pork',
    'nonstandard_exp',
    'selection_bias',
    'reverse_causation',
    "seq",
    'follow_up',
]


for dataset_path in redmeat_datasets:
    categorical_features_columns = []
    data = pd.read_csv(dataset_path)
    # This is Peng's formula for getting a covariate variable (exposure)
    data["linear_exposure"] = (data["b_0"] + data["b_1"]) / 2 - (data["a_0"] + data["a_1"]) / 2
    # TODO: ask Peng what I should use as SE given the transformation of target
    data["target"] = data["ln_effect"] / data["linear_exposure"]
    # normalization
    target_se = np.sqrt(data["target"].var())
    data["target"] = data["target"] / target_se
    data["se"] = (data["ln_se"] / (np.abs(data["linear_exposure"]) * target_se))
                  # data["se"] = 0.1 / np.abs(data["linear_exposure"])
                  # Data prep for the model

    for col in maybe_categorical_features_columns:
        if col in data.columns:
            data_col = data[col]
            if data_col.var() != 0 and data_col.min() * 1.1 < data_col.mean() < 0.9 * data_col.max():
                categorical_features_columns.append(col)

    groups_to_get = data['nid'].unique().tolist()  # [:4]
    data_short = data[main_features_columns + target_column + categorical_features_columns]
    data_short = data_short[data_short["nid"].isin(groups_to_get)]

    X = data_short[main_features_columns + categorical_features_columns].to_numpy()
    y = data_short[target_column].to_numpy().flatten()
    column_labels = [0, 4] + [3] * len(categorical_features_columns)
    X = np.vstack([column_labels, X])

    # plot coefficients trajectory
    figure = plt.figure(figsize=(12, 12))
    grid = plt.GridSpec(nrows=2, ncols=2)
    problem = LinearLMEProblem.from_x_y(X, y, random_intercept=True, add_group_as_categorical_feature=True)
    models = {}
    tbetas = np.zeros((problem.num_fixed_effects, problem.num_fixed_effects))
    losses = []
    selection_aics = []
    oracle = LinearLMEOracleRegularized(problem)
    for nnz_tbeta in range(len(categorical_features_columns)+1, 0, -1):
        for nnz_tgamma in range(nnz_tbeta, 0, -1):
            print(nnz_tbeta, nnz_tgamma)
            model = LinearLMESparseModel(nnz_tbeta=nnz_tbeta, nnz_tgamma=nnz_tgamma, n_iter_outer=20, initializer="EM",
                                         tol=1e-6)
            model.fit_problem(problem)
            y_pred = model.predict_problem(problem, use_sparse_coefficients=True)
            models[(nnz_tbeta, nnz_tgamma)] = (model, y_pred)
            if nnz_tbeta == nnz_tgamma:
                tbeta = model.coef_["tbeta"]
                tbetas[nnz_tbeta - 1, :] = tbeta
                losses.append(oracle.loss(**model.coef_))
                selection_aics.append(oracle.vaida2005aic(**model.coef_))
    # plot coefficients trajectory
    coefficients_plot = figure.add_subplot(grid[0, :])
    nnz_tbetas = range(1, len(categorical_features_columns) + 2, 1)
    colors = sns.color_palette("Set2", problem.num_fixed_effects)
    for i, feature in enumerate(["intercept"] + categorical_features_columns):
        plt.plot(nnz_tbetas, tbetas[:, i], label=feature, color=to_hex(colors[i]))
    coefficients_plot.legend()
    coefficients_plot.set_xlabel("NNZ beta")
    coefficients_plot.set_ylabel("Coefficients")

    # plot loss function and aics
    loss_plot = figure.add_subplot(grid[1, :])
    loss_plot.plot(nnz_tbetas, losses[::-1], label="Loss (R2)")
    loss_plot.plot(nnz_tbetas, selection_aics[::-1], label="AIC (R2)")
    loss_plot.set_xlabel("NNZ beta")
    loss_plot.set_xlabel("Loss")

    plt.savefig(figures_folder_path / f"{dataset_path.name}_feature_selection.png")

    # %% Calculate empirical gamma
    # mean_coefs = np.array(mean_coefs)
    # us = mean_coefs - model.coef_["beta"]
    # gamma = np.sum(us ** 2, axis=0) / problem.num_groups

    # %%
    # Shows distribution of target by nid
    # sns.boxplot(x=data_short["nid"], y=data_short["target"])
    # plt.show()
