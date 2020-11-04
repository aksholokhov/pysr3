import unittest
import time
from datetime import datetime
import pickle

import numpy as np
from scipy.linalg import block_diag
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score
from matplotlib import pyplot as plt

import pandas as pd

from skmixed.lme.problems import LinearLMEProblem
from skmixed.lme.oracles import LinearLMEOracle, LinearLMEOracleRegularized
from skmixed.lme.models import LinearLMESparseModel
from skmixed.lme.models import LassoLMEModel, LassoLMEModelFixedSelectivity

setup1_config = {
    "m": 30,
    "n_i": 5,
    "p": 9,
    "p_true": 2,  # excluding intercept: it's constrained to be 0
    "q": 4,
    "q_true": 3  # including intercept
}

setup2_config = {
    "m": 60,
    "n_i": 10,
    "p": 9,
    "p_true": 2,  # excluding intercept: it's constrained to be 0
    "q": 4,
    "q_true": 3  # including intercept
}

setups = [setup1_config, setup2_config]

def generate_krishna_problem(setup, seed=42) -> (LinearLMEProblem, dict):
    # This code creates two synthetic experiment setups from here:
    # https://pubmed.ncbi.nlm.nih.gov/20163404/
    m = setup["m"]
    n_i = setup["n_i"]
    p = setup["p"]
    q = setup["q"]

    np.random.seed(seed)
    num_groups = int(m/n_i)
    # we don't mention intercept here, it's defined to be random only when the LinearLMEProblem is constructed
    column_labels = [0] + [1]*p + [2]*(q-1) + [4]

    true_beta = np.array([1, 1] + [0]*(p-2))

    # true_gamma is not quite true in a sense that it's not diagonal by setup
    # but we still can use it for judging selection accuracy.
    # The only thing matters is zero on 4th position.
    true_gamma = np.array([1, 1, 1, 0])

    psi = np.array([
        [9, 4.8, 0.6, 0],
        [4.8, 4, 1, 0],
        [0.6, 1, 1, 0],
        [0, 0, 0, 0]
    ])

    x_full = None
    y_full = None

    true_parameters = {
        "beta": true_beta,
        "gamma": true_gamma,
        "random_effects": [],
        "errors": []
    }
    for i in range(num_groups):
        x = np.random.uniform(low=-2, high=2, size=(n_i, p))
        z = np.random.uniform(low=-2, high=2, size=(n_i, q))
        z[:, 0] = 1  # intercept
        u = np.random.multivariate_normal(mean=np.zeros(q), cov=psi)
        e = np.random.randn(n_i)
        y = x.dot(true_beta) + z.dot(u) + e
        group_column = np.array([i]*n_i).reshape((-1, 1))
        obs_std = np.ones(n_i).reshape((-1, 1))  # variances of measurement errors (we generated it from standard normal)

        group_block = np.hstack([group_column, x, z[:, 1:], obs_std])
        if x_full is None:
            x_full = group_block
            y_full = y
        else:
            x_full = np.vstack([x_full, group_block])
            y_full = np.concatenate([y_full, y])

        true_parameters["random_effects"].append(u)
        true_parameters["errors"].append(e)

    problem = LinearLMEProblem.from_x_y(x_full, y_full, columns_labels=column_labels, fixed_intercept=False, random_intercept=True)
    true_parameters["random_effects"] = np.array(true_parameters["random_effects"])
    true_parameters["errors"] = np.array(true_parameters["errors"])
    return problem, true_parameters

def run_experiment(seed_multiplier=1):
    model_parameters = {
        "initializer": "EM",
        "logger_keys": ('converged', 'loss',),
        "solver": "ip_combined",
        "tol_inner": 1e-3,
        "n_iter_inner": 3000,
    }

    log_columns = ["setup", "i", "CMF", "CMR", "TIME", "MSE", "MAB", "VAR"]

    for k, setup in enumerate(setups):
        print(f"Setup {k + 1}: m = {setup['m']}, n_i = {setup['n_i']} \n")
        # We first tune the parameters p and q via Mueller's IC and parametric bootstrap.
        log = pd.DataFrame(columns=log_columns)
        p_freqs = np.zeros(setup['p'])
        q_freqs = np.zeros(setup['q'])
        for i in range(200):
            # seeds can be whatever, just make sure they're different for each dataset
            problem, true_parameters = generate_krishna_problem(setup, seed=seed_multiplier*1000 * (k+1) + i)

            oracle = LinearLMEOracle(problem)
            best_aic = np.infty
            best_p = None
            best_q = None
            for p_t in range(1, setup["p"] + 1, 1):
                for q_t in range(1, setup["q"] + 1, 1):
                    model = LinearLMESparseModel(**model_parameters,
                                                 nnz_tbeta=p_t,
                                                 nnz_tgamma=q_t,
                                                 )
                    model.fit_problem(problem)
                    aic = oracle.muller2018ic(beta=model.coef_["tbeta"],
                                              gamma=model.coef_["tgamma"])
                    if aic < best_aic:
                        best_aic = aic
                        best_p = p_t
                        best_q = q_t
            p_freqs[best_p - 1] += 1
            q_freqs[best_q - 1] += 1

        p_tuned = int(1 + np.argmax(p_freqs))
        q_tuned = int(1 + np.argmax(q_freqs))
        print(f"Found optimal p={p_tuned} and q={q_tuned}")

        for i in range(200):
            # seed can be whatever, just make sure they're all different (and exactly the same to above)
            problem, true_parameters = generate_krishna_problem(setup, seed=seed_multiplier*1000 * (k+1) + i)
            x, y = problem.to_x_y()
            model = LinearLMESparseModel(**model_parameters,
                                                 nnz_tbeta=p_tuned,
                                                 nnz_tgamma=q_tuned,
                                                 )
            t0 = time.time()
            model.fit_problem(problem)
            fit_time = time.time() - t0
            y_pred = model.predict_problem(problem)

            if not np.isfinite(y_pred).all() or np.isnan(y_pred).any():
                raise Exception(f"{i}: y_pred is not finite. Coefs: {model.coef_},\n logger: {model.logger_.dict}")

            # explained_variance = explained_variance_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            # TODO: implement KL-discrepancy (requires implementing loss for a non-diagonal covariance matrix)

            coefficients = model.coef_
            maybe_tbeta = coefficients["tbeta"]
            maybe_tgamma = coefficients["tgamma"]
            fixed_effects_accuracy = accuracy_score(true_parameters["beta"] != 0, maybe_tbeta != 0)
            random_effects_accuracy = accuracy_score(true_parameters["gamma"] != 0, maybe_tgamma != 0)
            record = {
                "setup": k,
                "i": i,
                "CMF": fixed_effects_accuracy,
                "CMR": random_effects_accuracy,
                "TIME": fit_time,
                "MSE": mse
            }
            log = log.append(record, ignore_index=True)

        print(f"CMF={100*np.mean(log['CMF'] == 1):.0f} "
              f"CMR={100*np.mean(log['CMR'] == 1):.0f} "
              f"CM={100*np.mean((log['CMF'] == 1) & (log['CMR'] == 1)):.0f}\n"
              f"MSE={np.mean(log['MSE']):.2f}, TIME={np.mean(log['TIME'])}")

    return None


if __name__ == "__main__":
    # change seed multiplier to execute the experiment with
    # completely different randomly generated datasets
    run_experiment()



