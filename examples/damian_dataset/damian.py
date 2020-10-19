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

# np.seterr(all='raise', invalid='raise')

if __name__ == "__main__":
    base_directory = Path("/Users/aksh/Storage/repos/skmixed/examples/damian_dataset")
    figures_folder_path = base_directory / "figures"
    backups_folder_path = base_directory / "backups"
    dataset_path = base_directory / "liberal_data_file.csv"

    data = pd.read_csv(dataset_path)
    col_target = "log_effect_size"
    col_se = "log_effect_size_se"
    col_group = "author"
    group_to_id = {g: i for i, g in enumerate(data[col_group].unique())}
    data["group"] = data[col_group]
    data["group"] = data["group"].replace(group_to_id)
    col_group = "group"
    fixed_features_columns = ["time"]
    categorical_features_columns = [col for col in data.columns if col.startswith("cv_")]

    # data[col_target] *= 10
    # data[col_se] *= 10

    data["variance"] = data[col_se]**2

    # # experiment on variables scaling
    # data["target10"] = data[col_target]*10
    # data["se10"] = data[col_se]*10


    # Plotting simple fit without covariates
    X = data[[col_group, "variance", "time"]]
    column_labels = [0, 4, 1]
    X = np.vstack([column_labels, X])
    y = data[col_target]

    # Fitting the model
    problem = LinearLMEProblem.from_x_y(X, y, random_intercept=True)
    oracle = LinearLMEOracle(problem)

    # # Plotting simple fit without covariates
    # X2 = data[[col_group, "se10", "time"]]
    # column_labels = [0, 4, 1]
    # X2 = np.vstack([column_labels, X2])
    # y2 = data["target10"]
    #
    # # Fitting the model
    # problem2 = LinearLMEProblem.from_x_y(X2, y2, random_intercept=True)
    # oracle2 = LinearLMEOracle(problem2)

    model = LinearLMESparseModel(lb=0, lg=0,
                                 nnz_tbeta=2, nnz_tgamma=1,
                                 n_iter_outer=1,
                                 solver="ip",
                                 initializer="EM"
                                 )
    model.fit_problem(problem)
    y_pred = model.predict_problem(problem)

    # plot solution in the original space
    figure = plt.figure(figsize=(12, 12))
    grid = plt.GridSpec(nrows=2, ncols=2)
    # plot solutions
    solution_plot = figure.add_subplot(grid[0, 0])
    colors = sns.color_palette("Set2", problem.num_groups)
    mean_coefs = []
    max_time = data["time"].max()
    for i, (coef, (x, y, z, l)) in enumerate(zip(model.coef_["per_group_coefficients"], problem)):
        group_id = problem.group_labels[i]
        # coef[0] is intercept coef, coef[1:3] are 0s (group and std), coef[3:] are coefs for categorical covariates
        group_coefficient = coef[1]
        time = x[:, 1]
        color = to_hex(colors[i])
        solution_plot.scatter(time, y, label=group_id, c=color)
        solution_plot.errorbar(time, y, yerr=np.sqrt(l), c=color, fmt="none")
        solution_plot.plot([0, max_time], [coef[0], coef[0] + coef[3] * max_time], c=color)
    # solution_plot.legend()
    solution_plot.set_title(f"Solution, gamma = {model.coef_['gamma']}, beta = {model.coef_['beta']}")
    solution_plot.set_xlabel("Time")
    solution_plot.set_ylabel("Target")

    # plot residues
    residuals_plot = figure.add_subplot(grid[0, 1])
    data["predictions"] = y_pred
    data["residuals"] = (data[col_target] - data["predictions"])
    sns.swarmplot(x="group", y="residuals", data=data, ax=residuals_plot, palette=colors)
    residuals_plot.set_xticklabels(range(problem.num_groups))
    residuals_plot.set_title("residuals")
    residuals_plot.set_xlabel("groups")

    # plot loss functions and information criteria
    oracle = LinearLMEOracle(problem)
    demarginalized_loss_plot = figure.add_subplot(grid[1, 1])
    loss_plot = figure.add_subplot(grid[1, 0])
    # aic_plot = figure.add_subplot(grid[1, 1])
    demarginalized_losses = []
    losses = []
    aics = []
    gammas = np.linspace(0, max(5, 2*model.coef_['gamma'][0]), 100)

    for g in gammas:
        current_gamma = np.array([g])
        current_beta = oracle.optimal_beta(current_gamma)
        losses.append(oracle.loss(current_beta, current_gamma))
        aics.append(oracle.vaida2005aic(current_beta, current_gamma))
        demarginalized_losses.append(oracle.demarginalized_loss(current_beta, current_gamma))
    loss_plot.plot(gammas, losses)
    demarginalized_loss_plot.plot(gammas, demarginalized_losses)

    # aic_plot.plot(gammas, aics)
    loss_plot.set_title("Loss depending on intercept variance (gamma)")
    loss_plot.set_xlabel("Variance for intercept's RE (gamma)")
    loss_plot.set_ylabel("Loss function value")
    demarginalized_loss_plot.set_title("Demarginalized loss depending on gamma")
    demarginalized_loss_plot.set_xlabel("Variance for intercept's RE (gamma)")
    demarginalized_loss_plot.set_ylabel("Loss function value")
    # aic_plot.set_title("AIC")

    plt.savefig(figures_folder_path / f"{dataset_path.name}_intercept_only.png")
    plt.close()

    # plot coefficients trajectory

    X = data[[col_group, "variance", "time"] + categorical_features_columns]
    column_labels = [0, 4, 1] + [3] * len(categorical_features_columns)
    X = np.vstack([column_labels, X])
    y = data[col_target]

    figure = plt.figure(figsize=(12, 30))
    grid = plt.GridSpec(nrows=5, ncols=2)
    problem = LinearLMEProblem.from_x_y(X, y, random_intercept=True)
    models = {}
    tbetas = np.zeros((problem.num_fixed_effects-1, problem.num_fixed_effects))
    tgammas = np.zeros((problem.num_random_effects-0, problem.num_random_effects))
    losses = []
    losses_dense = []
    losses_w = []
    selection_aics = []
    selection_aics_dense = []
    selection_aics_w = []

    # intercept and time do not participate in the selection process
    participation_in_selection = np.array([False, False] + [True] * len(categorical_features_columns))

    trials = 30
    from scipy.special import factorial

    for nnz_tbeta in range(len(categorical_features_columns) + 2, 1, -1):
        for nnz_tgamma in range(nnz_tbeta-1, nnz_tbeta - 2, -1):
            #s = factorial(len(categorical_features_columns)) / (factorial(nnz_tbeta)*factorial(len(categorical_features_columns) - nnz_tbeta))
            # exhaustive search
            # for j in range(min(s, trials)):
            #     features_to_take = sorted(np.random.choice(range(len(categorical_features_columns)))) + 2
            #     X = data[[col_group, col_se, "time"] + categorical_features_columns]
            #     column_labels = [0, 4, 1] + [3] * len(categorical_features_columns)
            #     X = np.vstack([column_labels, X])
            #     y = data[col_target]

            # model fit
            model = LinearLMESparseModel(nnz_tbeta=nnz_tbeta,
                                         nnz_tgamma=nnz_tgamma,
                                         n_iter_outer=20,
                                         tol_inner=1e-4,
                                         tol_outer=1e-4,
                                         solver="ip",
                                         initializer="EM",
                                         participation_in_selection=participation_in_selection)
            model.fit_problem(problem)
            y_pred = model.predict_problem(problem, use_sparse_coefficients=True)
            models[(nnz_tbeta, nnz_tgamma)] = (model, )
            if nnz_tbeta-1 == nnz_tgamma:
                tbetas[nnz_tbeta - 2, :] = model.coef_["tbeta"]
                tgammas[nnz_tgamma - 1, :] = model.coef_["tgamma"]
                oracle = LinearLMEOracleRegularized(problem, lb=0, lg=0, nnz_tbeta=nnz_tbeta, nnz_tgamma=nnz_tgamma)
                losses.append(oracle.loss(beta=model.coef_["tbeta"],
                                          gamma=model.coef_["tgamma"],
                                          tbeta=model.coef_["tbeta"],
                                          tgamma=model.coef_["tgamma"]))
                selection_aics.append(oracle.vaida2005aic(beta=model.coef_["beta"],
                                          gamma=model.coef_["gamma"],
                                          tbeta=model.coef_["tbeta"],
                                          tgamma=model.coef_["tgamma"]))
            print(f"{nnz_tbeta}-{nnz_tgamma} ", end='')
    print('\n')
    # with open(backups_folder_path / f"{damian_dataset_path.name}_model_backup_{datetime.datetime.now()}", 'wb') as f:
    #     pickle.dump((dataset_path, problem, categorical_features_columns, models), f)
    # plot coefficients trajectory
    colors = sns.color_palette("Set2", problem.num_fixed_effects)

    betas_plot = figure.add_subplot(grid[0, :])
    inclusion_betas_plot = figure.add_subplot(grid[1, :])
    nnz_tbetas = np.array(range(2, len(categorical_features_columns) + 3, 1))
    beta_features_labels = []
    for i, feature in enumerate(["intercept", "time"] + categorical_features_columns):
        betas_plot.plot(nnz_tbetas, tbetas[:, i], label=feature, color=to_hex(colors[i]))
        inclusion_betas = np.copy(tbetas[:, i])
        idx_zero_betas = inclusion_betas == 0
        inclusion_betas[idx_zero_betas] = None
        inclusion_betas[~idx_zero_betas] = i
        inclusion_betas_plot.plot(nnz_tbetas, inclusion_betas, label=feature, color=to_hex(colors[i]))
        beta_features_labels.append(feature)


    betas_plot.set_xticks(nnz_tbetas)
    inclusion_betas_plot.set_xticks(nnz_tbetas)
    inclusion_betas_plot.set_yticks(range(len(beta_features_labels)))
    inclusion_betas_plot.set_yticklabels(beta_features_labels)

    betas_plot.legend()
    betas_plot.set_xlabel(r"$\|\beta_0\|$: maximum number of non-zero fixed effects allowed in the model.")
    betas_plot.set_ylabel(r"$\beta$: fixed effects")
    betas_plot.set_title(f"{dataset_path.name}: optimal coefficients for fixed effects depending on maximum non-zero coefficients allowed.")
    # plot gammas trajectory
    gammas_plot = figure.add_subplot(grid[2, :])
    inclusion_gammas_plot = figure.add_subplot(grid[3, :])
    nnz_tgammas = np.array(range(1, len(categorical_features_columns) + 2, 1))
    gamma_features_labels = []
    for i, feature in enumerate(["intercept"] + categorical_features_columns):
        color = to_hex(colors[i+1]) if i > 0 else to_hex(colors[i])
        gammas_plot.plot(nnz_tgammas, tgammas[:, i], '--', label=feature, color=color)
        inclusion_gammas = np.copy(tgammas[:, i])
        idx_zero_gammas = inclusion_gammas == 0
        inclusion_gammas[idx_zero_gammas] = None
        inclusion_gammas[~idx_zero_gammas] = i
        inclusion_gammas_plot.plot(nnz_tgammas, inclusion_gammas, '--', label=feature, color=color)
        gamma_features_labels.append(feature)

    gammas_plot.legend()
    gammas_plot.set_xlabel(r"$\|\gamma\|_0$: maximum number of non-zero random effects allowed in the model.")
    gammas_plot.set_ylabel(r"$\gamma$: variance of random effects")
    gammas_plot.set_title(f"{dataset_path.name}: optimal variances of random effects depending on maximum non-zero coefficients allowed.")

    gammas_plot.set_xticks(nnz_tgammas)
    inclusion_gammas_plot.set_xlabel(r"*the time is constrained to be always included as a fixed effect, so the procedure asks to choose $i$ fixed effects and $i-1$ random effects (out of those $i$ fixed effects).")
    inclusion_gammas_plot.set_xticks(nnz_tgammas)
    inclusion_gammas_plot.set_yticks(range(len(gamma_features_labels)))
    inclusion_gammas_plot.set_yticklabels(gamma_features_labels)

    # plot loss function and aics
    loss_plot = figure.add_subplot(grid[4, 0])
    loss_plot.set_title("Loss")
    loss_plot.plot(nnz_tbetas, losses[::-1], label="Loss (R2)")
    loss_plot.set_xlabel("NNZ beta")
    loss_plot.set_ylabel("Loss")
    loss_plot.legend()
    aics_plot = figure.add_subplot(grid[4, 1])
    aics_plot.set_title("AICs")
    selection_aics = np.array(selection_aics[::-1])
    aics_plot.plot(nnz_tbetas, selection_aics, label="AIC (R2)")
    argmin_aic = np.argmin(selection_aics)
    aics_plot.scatter(nnz_tbetas[argmin_aic], selection_aics[argmin_aic], s=80, facecolors='none', edgecolors='r')
    inclusion_betas_plot.scatter([nnz_tbetas[argmin_aic]]*len(beta_features_labels), [None if tbetas[argmin_aic, i] == 0 else i for i, f in enumerate(beta_features_labels)], s=80, facecolors='none', edgecolors='r')
    inclusion_gammas_plot.scatter([nnz_tgammas[argmin_aic]]*len(gamma_features_labels), [None if tgammas[argmin_aic, i] == 0 else i for i, f in enumerate(gamma_features_labels)], s=80, facecolors='none', edgecolors='r')
    aics_plot.set_xlabel("NNZ beta")
    aics_plot.set_ylabel("AIC")
    aics_plot.legend()
    aics_plot.set_xticks(nnz_tbetas)
    loss_plot.set_xticks(nnz_tbetas)
    plt.savefig(figures_folder_path / f"feature_selection.png")
    plt.close()