from pathlib import Path

from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score
import pandas as pd
import numpy as np
import pickle
import datetime
from pandas.api.types import is_numeric_dtype

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import seaborn as sns

from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle, LinearLMEOracleRegularized, LinearLMEOracleW
from skmixed.lme.models import LinearLMESparseModel

from skmixed.lme.trees import Tree, Forest

np.seterr(all='raise', invalid='raise')

# %%
figures_folder_path = Path("figures")
backups_folder_path = Path("backups")
datasets_folder_path = Path("/Users/aksh/Storage/repos/skmixed/datasets/evidence_score_data/")
redmeat_datasets = [f for f in datasets_folder_path.iterdir() if str(f.name) != ".DS_Store"]

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
#%%

categorical_features_columns = []
# %% generate intercept only solutions
for dataset_path in redmeat_datasets[:3]:
    print(dataset_path)

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

    # Features which participate in selection (all except the intercept)
    participation_in_selection = np.array([False] + [True] * len(categorical_features_columns))

    # Fitting the model
    problem = LinearLMEProblem.from_x_y(X, y, random_intercept=True, add_group_as_categorical_feature=True)
    model = LinearLMESparseModel(lb=0, lg=0,
                                 nnz_tbeta=1, nnz_tgamma=1,
                                 n_iter_outer=1,
                                 initializer=None,
                                 tol=1e-5,
                                 participation_in_selection=participation_in_selection)
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
    plt.close()

# %% plot feature selection plots
# print("feature selection starts")
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


for dataset_path in redmeat_datasets[1:3]:
    print(f"{dataset_path.name}")
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

    for col in data.columns:
        if col in ["field_citation_value",
                   "location",
                   "seq",
                   "washout_years",
                   "nid","b_0", "b_1", "a_0", "a_1", "ln_effect", "ln_se", "se", "linear_exposure", "target"]:
            continue
        data_col = data[col]
        if not is_numeric_dtype(data_col) or not np.isfinite(data_col).all() or np.isnan(data_col).any():
            continue
        if data_col.var() != 0 and data_col.min() * 1.1 < data_col.mean() < 0.9 * data_col.max():
            categorical_features_columns.append(col)

    groups_to_get = data['nid'].unique().tolist()  # [:4]
    data_short = data[main_features_columns + target_column + categorical_features_columns]
    data_short = data_short[data_short["nid"].isin(groups_to_get)]

    X = data_short[main_features_columns + categorical_features_columns].to_numpy()
    y = data_short[target_column].to_numpy().flatten()
    column_labels = [0, 4] + [3] * len(categorical_features_columns)
    X = np.vstack([column_labels, X])

    # Features which participate in selection (all except the intercept)
    participation_in_selection = np.array([False] + [True] * len(categorical_features_columns))

    # plot coefficients trajectory
    figure = plt.figure(figsize=(12, 18))
    grid = plt.GridSpec(nrows=3, ncols=2)
    problem = LinearLMEProblem.from_x_y(X, y, random_intercept=True, add_group_as_categorical_feature=True)
    models = {}
    tbetas = np.zeros((problem.num_fixed_effects, problem.num_fixed_effects))
    tbetas_w = np.zeros((problem.num_fixed_effects, problem.num_fixed_effects))
    tgammas = np.zeros((problem.num_random_effects, problem.num_random_effects))
    losses = []
    losses_dense = []
    losses_w = []
    selection_aics = []
    selection_aics_dense = []
    selection_aics_w = []

    for nnz_tbeta in range(len(categorical_features_columns)+1, 0, -1):
        for nnz_tgamma in range(nnz_tbeta, nnz_tbeta - 1, -1):
            model = LinearLMESparseModel(nnz_tbeta=nnz_tbeta, nnz_tgamma=nnz_tgamma, n_iter_outer=20, initializer=None,
                                         tol=1e-5, tol_outer=1e-5, participation_in_selection=participation_in_selection)
            model_w = LinearLMESparseModel(nnz_tbeta=nnz_tbeta, nnz_tgamma=nnz_tgamma, n_iter_outer=20, initializer=None,
                                           regularization_type="loss-weighted", tol=1e-5, tol_outer=1e-5)
            model.fit_problem(problem)
            #model_w.fit_problem(problem)
            y_pred = model.predict_problem(problem, use_sparse_coefficients=True)
            #y_pred_w = model_w.predict_problem(problem, use_sparse_coefficients=True)
            models[(nnz_tbeta, nnz_tgamma)] = (model, model_w)
            if nnz_tbeta == nnz_tgamma:
                tbetas[nnz_tbeta - 1, :] = model.coef_["tbeta"]
                tgammas[nnz_tgamma - 1, :] = model.coef_["tgamma"]
                #tbetas_w[nnz_tbeta - 1, :] = tbeta
                oracle = LinearLMEOracleRegularized(problem, lb=0, lg=0, nnz_tbeta=nnz_tbeta, nnz_tgamma=nnz_tgamma)
                losses.append(oracle.loss(beta=model.coef_["tbeta"],
                                          gamma=model.coef_["tgamma"],
                                          tbeta=model.coef_["tbeta"],
                                          tgamma=model.coef_["tgamma"]))
                # losses_dense.append(oracle.loss(beta=model.coef_["beta"],
                #                           gamma=model.coef_["gamma"],
                #                           tbeta=model.coef_["beta"],
                #                           tgamma=model.coef_["gamma"]))
                #losses_w.append(oracle.loss(**model_w.coef_))
                selection_aics.append(oracle.vaida2005aic(beta=model.coef_["beta"],
                                          gamma=model.coef_["gamma"],
                                          tbeta=model.coef_["tbeta"],
                                          tgamma=model.coef_["tgamma"]))
                # selection_aics_dense.append(oracle.vaida2005aic(beta=model.coef_["beta"],
                #                                           gamma=model.coef_["gamma"],
                #                                           tbeta=model.coef_["beta"],
                #                                           tgamma=model.coef_["gamma"]))
                #selection_aics_w.append(oracle.vaida2005aic(**model_w.coef_))
            print(f"{nnz_tbeta}-{nnz_tgamma} ", end='')
    print('\n')
    with open(backups_folder_path / f"{dataset_path.name}_model_backup_{datetime.datetime.now()}", 'wb') as f:
        pickle.dump((dataset_path, problem, categorical_features_columns, models), f)
    # plot coefficients trajectory
    coefficients_plot = figure.add_subplot(grid[0, :])
    nnz_tbetas = range(1, len(categorical_features_columns) + 2, 1)
    colors = sns.color_palette("Set2", problem.num_fixed_effects)
    for i, feature in enumerate(["intercept"] + categorical_features_columns):
        coefficients_plot.plot(nnz_tbetas, tbetas[:, i], label=feature, color=to_hex(colors[i]))
 #       plt.plot(nnz_tbetas, tbetas_w[:, i], style=":", color=to_hex(colors[i]))
    coefficients_plot.legend()
    coefficients_plot.set_xlabel("NNZ beta")
    coefficients_plot.set_ylabel("Coefficients")
    coefficients_plot.set_title(f"{dataset_path.name}: feature selection for fixed effects")
    # plot gammas trajectory
    gammas_plot = figure.add_subplot(grid[1, :])
    nnz_tgammas = range(1, len(categorical_features_columns) + 2, 1)
    for i, feature in enumerate(["intercept"] + categorical_features_columns):
        gammas_plot.plot(nnz_tgammas, tgammas[:, i], '--', label=feature, color=to_hex(colors[i]))
    gammas_plot.legend()
    gammas_plot.set_xlabel("NNZ gamma")
    gammas_plot.set_ylabel("Coefficients")
    gammas_plot.set_title(f"{dataset_path.name}: feature selection for random effects")

    # plot loss function and aics
    loss_plot = figure.add_subplot(grid[2, 0])
    loss_plot.set_title("Loss")
    loss_plot.plot(nnz_tbetas, losses[::-1], label="Loss (R2)")
    #loss_plot.plot(nnz_tbetas, losses_dense[::-1], label="Loss dense (R2)")
 #   loss_plot.plot(nnz_tbetas, losses_w[::-1], label="Loss (W)")
    loss_plot.set_xlabel("NNZ beta")
    loss_plot.set_ylabel("Loss")
    loss_plot.legend()
    aics_plot = figure.add_subplot(grid[2, 1])
    aics_plot.set_title("AICs")
    aics_plot.plot(nnz_tbetas, selection_aics[::-1], label="AIC (R2)")
    # aics_plot.plot(nnz_tbetas, selection_aics_dense[::-1], label="AIC dense (R2)")
#    aics_plot.plot(nnz_tbetas, selection_aics_w[::-1], label="AIC (W)")
    aics_plot.set_xlabel("NNZ beta")
    aics_plot.set_ylabel("AIC")
    aics_plot.legend()
    plt.savefig(figures_folder_path / f"{dataset_path.name}_feature_selection.png")
    plt.close()

    # %% Calculate empirical gamma
    # mean_coefs = np.array(mean_coefs)
    # us = mean_coefs - model.coef_["beta"]
    # gamma = np.sum(us ** 2, axis=0) / problem.num_groups

    # %%
    # Shows distribution of target by nid
    # sns.boxplot(x=data_short["nid"], y=data_short["target"])
    # plt.show()
 #%% Models analysis
# with open(Path("/Users/aksh/Storage/repos/skmixed/examples") / backups_folder_path / "redmeat_allcausemortality.csv_model_backup", 'rb') as f:
#     dataset_path, problem, categorical_features_columns, models = pickle.load(f)
#
#
#
# for nnz_tbeta in range(len(categorical_features_columns) + 1, 0, -1):
#     for nnz_tgamma in range(nnz_tbeta, 0, -1):
#         model, _ = models[(nnz_tbeta, nnz_tgamma)]
#         oracle = LinearLMEOracleRegularized()