from pathlib import Path

import pandas as pd
import numpy as np

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import seaborn as sns

from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle, LinearLMEOracleRegularized
from skmixed.lme.models import LinearLMESparseModel

from examples.general.settings import thesis_proposal_figures, thesis_presentation_figures, presentation_background_color

# np.seterr(all='raise', invalid='raise')

# based on the expert's prior knowledge and their experiences
# during previous rounds of GBD
historic_significance = {
    "cv_symptoms": 0,
    "cv_unadjusted": 1,
    "cv_b_parent_only": 1,
    "cv_or": 0,
    "cv_multi_reg": 1,
    "cv_low_threshold_bullying": 1,
    "cv_anx": 1,
    "percent_female": 1,
    "cv_selection_bias": 1,
    "cv_child_baseline": 0,
    "intercept": 1,
    "time": 1,
    "cv_baseline_adjust": 0
}

def plot_selection(ax, x, y_true, y_pred):
    true_pos = (y_true == True) & (y_pred == True)
    true_neg = (y_true == False) & (y_pred == False)
    false_pos = (y_true == False) & (y_pred == True)
    false_neg = (y_true == True) & (y_pred == False)

    ax.scatter([x] * len(y_true),
                    [None if t == False else i for i, t in
                     enumerate(true_pos)], s=80, facecolors='none', edgecolors='g', label="True Positive")
    ax.scatter([x] * len(y_true),
               [None if t == False else i for i, t in
                enumerate(false_pos)], s=80, facecolors='none', edgecolors='r', label="False Positive")
    ax.scatter([x] * len(y_true),
               [None if t == False else i for i, t in
                enumerate(true_neg)], marker='X', s=80, facecolors='none', edgecolors='g', label="True Negative")
    ax.scatter([x] * len(y_true),
               [None if t == False else i for i, t in
                enumerate(false_neg)], marker='X', s=80, facecolors='none', edgecolors='r', label="False Negative")
    ax.legend()


def generate_bullying_experiment(presentation=False, pres_one_pic=True):
    base_directory = Path("/Users/aksh/Storage/repos/skmixed/examples/bullying")
    dataset_path = base_directory / "bullying_data.csv"

    data = pd.read_csv(dataset_path)
    col_target = "log_effect_size"
    col_se = "log_effect_size_se"
    col_group = "cohort"
    group_to_id = {g: i for i, g in enumerate(data[col_group].unique())}
    data["group"] = data[col_group]
    data["group"] = data["group"].replace(group_to_id)
    col_group = "group"
    categorical_features_columns = [col for col in data.columns if col.startswith("cv_")] + [
        "percent_female",
    ]


    data["variance"] = data[col_se] ** 2

    X = data[[col_group, "variance", "time"]]
    column_labels = [0, 4, 1]
    X = np.vstack([column_labels, X])
    y = data[col_target]

    # Fitting the model
    problem = LinearLMEProblem.from_x_y(X, y, random_intercept=True)

    model = LinearLMESparseModel(lb=0, lg=0,
                                 nnz_tbeta=2, nnz_tgamma=1,
                                 n_iter_outer=1,
                                 solver="ip",
                                 initializer="EM"
                                 )
    model.fit_problem(problem)
    y_pred = model.predict_problem(problem)

    # plot solution in the original space
    figure = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(nrows=1, ncols=2)
    # plot solutions
    solution_plot = figure.add_subplot(grid[0, 0])
    colors = sns.color_palette("husl", problem.num_groups)
    max_time = data["time"].max()
    for i, (coef, (x, y, z, l)) in enumerate(zip(model.coef_["per_group_coefficients"], problem)):
        group_id = problem.group_labels[i]
        time = x[:, 1]
        color = to_hex(colors[i])
        solution_plot.scatter(time, y, label=group_id, c=color)
        solution_plot.errorbar(time, y, yerr=np.sqrt(l), c=color, fmt="none")
        solution_plot.plot([0, max_time], [coef[0], coef[0] + coef[3] * max_time], c=color)
    # solution_plot.legend()
    solution_plot.set_title(
        "Solution: " + r"$\beta$" + f" = [{model.coef_['beta'][0]:.2f}, {model.coef_['beta'][1]:.2f}], " + r"$\gamma$" + f" = [{model.coef_['gamma'][0]:.2f}]")
    solution_plot.set_xlabel("Time")
    solution_plot.set_ylabel("Target")


    # plot loss functions and information criteria
    oracle = LinearLMEOracle(problem)
    loss_plot = figure.add_subplot(grid[0, 1])
    losses = []
    aics = []
    gammas = np.linspace(0, max(5, 2 * model.coef_['gamma'][0]), 100)

    for g in gammas:
        current_gamma = np.array([g])
        current_beta = oracle.optimal_beta(current_gamma)
        losses.append(oracle.loss(current_beta, current_gamma))
        aics.append(oracle.vaida2005aic(current_beta, current_gamma))
    loss_plot.plot(gammas, losses)
    loss_plot.set_title("Loss depending on intercept variance (gamma)")
    loss_plot.set_xlabel("Variance for intercept's RE (gamma)")
    loss_plot.set_ylabel("Loss function value")

    if presentation:
        plt.savefig(thesis_presentation_figures / f"{dataset_path.name}_intercept_only.pdf")
    else:
        plt.savefig(thesis_proposal_figures / f"{dataset_path.name}_intercept_only.pdf")
    plt.close()

    # plot coefficients trajectory

    X = data[[col_group, "variance", "time"] + categorical_features_columns]
    column_labels = [0, 4, 1] + [3] * len(categorical_features_columns)
    X = np.vstack([column_labels, X])
    y = data[col_target]

    if presentation:
        if pres_one_pic:
            figure1 = plt.figure(figsize=(12, 12))
            grid1 = plt.GridSpec(nrows=3, ncols=2)
            loss_plot = figure1.add_subplot(grid1[0, 0])
            aics_plot = figure1.add_subplot(grid1[0, 1])
            betas_plot = figure1.add_subplot(grid1[1, :])
            gammas_plot = figure1.add_subplot(grid1[2, :])

            figure2 = plt.figure(figsize=(12, 9))
            grid2= plt.GridSpec(nrows=2, ncols=2)
            inclusion_betas_plot = figure2.add_subplot(grid2[0, :])
            inclusion_gammas_plot = figure2.add_subplot(grid2[1, :])


        else:
            figure1 = plt.figure(figsize=(12, 9))
            grid1 = plt.GridSpec(nrows=2, ncols=2)
            loss_plot = figure1.add_subplot(grid1[0, 0])
            aics_plot = figure1.add_subplot(grid1[0, 1])
            inclusion_betas_plot = figure1.add_subplot(grid1[1, :])

            figure2 = plt.figure(figsize=(12, 9))
            grid2 = plt.GridSpec(nrows=2, ncols=2)
            loss_plot_2 = figure2.add_subplot(grid2[0, 0])
            aics_plot_2 = figure2.add_subplot(grid2[0, 1])
            inclusion_gammas_plot = figure2.add_subplot(grid2[1, :])

            figure3 = plt.figure(figsize=(12, 12))
            grid3 = plt.GridSpec(nrows=2, ncols=2)
            betas_plot = figure3.add_subplot(grid3[0, :])
            gammas_plot = figure3.add_subplot(grid3[1, :])

    else:
        figure1 = plt.figure(figsize=(12, 18))
        grid1 = plt.GridSpec(nrows=3, ncols=2)
        loss_plot = figure1.add_subplot(grid1[0, 0])
        aics_plot = figure1.add_subplot(grid1[0, 1])
        betas_plot = figure1.add_subplot(grid1[1, :])
        inclusion_betas_plot = figure1.add_subplot(grid1[2, :])

        figure2 = plt.figure(figsize=(12, 12))
        grid2 = plt.GridSpec(nrows=2, ncols=2)
        gammas_plot = figure2.add_subplot(grid2[0, :])
        inclusion_gammas_plot = figure2.add_subplot(grid2[1, :])

    problem = LinearLMEProblem.from_x_y(X, y, random_intercept=True)
    models = {}
    tbetas = np.zeros((problem.num_fixed_effects - 1, problem.num_fixed_effects))
    tgammas = np.zeros((problem.num_random_effects - 0, problem.num_random_effects))
    losses = []
    selection_aics = []

    # intercept and time do not participate in the selection process
    participation_in_selection = np.array([False, False] + [True] * len(categorical_features_columns))


    for nnz_tbeta in range(len(categorical_features_columns) + 2, 1, -1):
        for nnz_tgamma in range(nnz_tbeta - 1, nnz_tbeta - 2, -1):
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
            models[(nnz_tbeta, nnz_tgamma)] = (model,)
            if nnz_tbeta - 1 == nnz_tgamma:
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

    colors = sns.color_palette("husl", problem.num_fixed_effects)

    nnz_tbetas = np.array(range(2, len(categorical_features_columns) + 3, 1))
    beta_features_labels = []
    for i, feature in enumerate(["intercept", "time"] + categorical_features_columns):
        betas_plot.plot(nnz_tbetas, tbetas[:, i], label=feature, color=to_hex(colors[i]))
        inclusion_betas = np.copy(tbetas[:, i])
        idx_zero_betas = inclusion_betas == 0
        inclusion_betas[idx_zero_betas] = None
        inclusion_betas[~idx_zero_betas] = i
        inclusion_betas_plot.plot(nnz_tbetas, inclusion_betas, color=to_hex(colors[i]))
        beta_features_labels.append(feature)

    betas_plot.set_xticks(nnz_tbetas)
    inclusion_betas_plot.set_xticks(nnz_tbetas)
    inclusion_betas_plot.set_yticks(range(len(beta_features_labels)))
    inclusion_betas_plot.set_yticklabels(beta_features_labels)

    betas_plot.legend()
    inclusion_betas_plot.set_xlabel(r"$\|\beta\|_0$: maximum number of non-zero fixed effects allowed in the model.")
    betas_plot.set_xlabel(r"$\|\beta\|_0$: maximum number of non-zero fixed effects allowed in the model.")
    betas_plot.set_ylabel(r"$\beta$: fixed effects")
    betas_plot.set_title(
        f"{dataset_path.name}: optimal coefficients for fixed effects depending on maximum non-zero coefficients allowed.")
    # plot gammas trajectory

    # plot loss function and aics
    loss_plot.set_title("Loss")
    loss_plot.plot(nnz_tbetas, losses[::-1], label="Loss (R2)")
    loss_plot.set_xlabel(r"$\|\beta\|_0$ -- number of NNZ coefficients")
    loss_plot.set_ylabel("Loss")
    loss_plot.legend()
    loss_plot.set_xticks(nnz_tbetas)
    if presentation and not pres_one_pic:
        loss_plot_2.set_title("Loss")
        loss_plot_2.plot(nnz_tbetas, losses[::-1], label="Loss (R2)")
        loss_plot_2.set_xlabel(r"$\|\beta\|_0$ -- number of NNZ coefficients")
        loss_plot_2.set_ylabel("Loss")
        loss_plot_2.legend()
        loss_plot_2.set_xticks(nnz_tbetas)

    selection_aics = np.array(selection_aics[::-1])
    argmin_aic = np.argmin(selection_aics)
    aics_plot.plot(nnz_tbetas, selection_aics, label="AIC (R2)")
    aics_plot.scatter(nnz_tbetas[argmin_aic], selection_aics[argmin_aic], s=80, facecolors='none', edgecolors='r')
    aics_plot.set_xlabel(r"$\|\beta\|_0$ -- number of NNZ coefficients")
    aics_plot.set_ylabel("AIC")
    aics_plot.legend()
    aics_plot.set_xticks(nnz_tbetas)
    aics_plot.set_title("AIC")
    if presentation and not pres_one_pic:
        aics_plot_2.plot(nnz_tbetas, selection_aics, label="AIC (R2)")
        aics_plot_2.scatter(nnz_tbetas[argmin_aic], selection_aics[argmin_aic], s=80, facecolors='none', edgecolors='r')
        aics_plot_2.set_xlabel(r"$\|\beta\|_0$ -- number of NNZ coefficients")
        aics_plot_2.set_ylabel("AIC")
        aics_plot_2.legend()
        aics_plot_2.set_xticks(nnz_tbetas)
        aics_plot_2.set_title("AIC")

    beta_historic_significance = np.array([bool(historic_significance[feature]) for feature in beta_features_labels])
    beta_predicted_significance = np.array([tbetas[argmin_aic, i] != 0 for i, f in
                                            enumerate(beta_features_labels)])
    plot_selection(inclusion_betas_plot, nnz_tbetas[argmin_aic], beta_historic_significance,
                   beta_predicted_significance)

    if presentation:
        if pres_one_pic:
            figure1.savefig(thesis_presentation_figures / f"{dataset_path.name}_beta_gamma_aics.pdf", facecolor = presentation_background_color)
        else:
            figure1.savefig(thesis_presentation_figures / f"{dataset_path.name}_fixed_feature_selection.pdf", facecolor = presentation_background_color)
    else:
        figure1.savefig(thesis_proposal_figures / f"{dataset_path.name}_fixed_feature_selection.png")

    ## Random feature selection plot

    nnz_tgammas = np.array(range(1, len(categorical_features_columns) + 2, 1))
    gamma_features_labels = []
    for i, feature in enumerate(["intercept"] + categorical_features_columns):
        color = to_hex(colors[i + 1]) if i > 0 else to_hex(colors[i])
        gammas_plot.plot(nnz_tgammas, tgammas[:, i], '--', label=feature, color=color)
        inclusion_gammas = np.copy(tgammas[:, i])
        idx_zero_gammas = inclusion_gammas == 0
        inclusion_gammas[idx_zero_gammas] = None
        inclusion_gammas[~idx_zero_gammas] = i
        inclusion_gammas_plot.plot(nnz_tgammas, inclusion_gammas, '--', color=color)
        gamma_features_labels.append(feature)

    gammas_plot.legend()
    gammas_plot.set_xlabel(r"$\|\gamma\|_0$: maximum number of non-zero random effects allowed in the model.")
    gammas_plot.set_ylabel(r"$\gamma$: variance of random effects")
    gammas_plot.set_title(
        f"{dataset_path.name}: optimal variances of random effects depending on maximum non-zero coefficients allowed.")

    gammas_plot.set_xticks(nnz_tgammas)
    if presentation:
        inclusion_gammas_plot.set_xlabel(r"$\|\gamma\|_0$: maximum number of non-zero random effects allowed in the model.")
    else:
        inclusion_gammas_plot.set_xlabel(
            r"*the time is constrained to be always included as a fixed effect, so the procedure asks to choose $i$ fixed effects and $i-1$ random effects (out of those $i$ fixed effects).")
    inclusion_gammas_plot.set_xticks(nnz_tgammas)
    inclusion_gammas_plot.set_yticks(range(len(gamma_features_labels)))
    inclusion_gammas_plot.set_yticklabels(gamma_features_labels)

    gamma_historic_significance = np.array([bool(historic_significance[feature]) for feature in gamma_features_labels])
    gamma_predicted_significance = np.array([tgammas[argmin_aic, i] != 0 for i, f in
                                             enumerate(gamma_features_labels)])
    plot_selection(inclusion_gammas_plot, nnz_tgammas[argmin_aic], gamma_historic_significance,
                   gamma_predicted_significance)

    if presentation:
        if pres_one_pic:
            figure2.savefig(thesis_presentation_figures / f"{dataset_path.name}_inclusion.pdf", facecolor = presentation_background_color)
        else:
            figure2.savefig(thesis_presentation_figures / f"{dataset_path.name}_random_feature_selection.pdf", facecolor = presentation_background_color)
    else:
        figure2.savefig(thesis_proposal_figures / f"{dataset_path.name}_random_feature_selection.pdf")
    plt.close()


if __name__ == "__main__":
    generate_bullying_experiment(presentation=True, pres_one_pic=True)