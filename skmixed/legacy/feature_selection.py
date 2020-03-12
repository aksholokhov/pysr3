from skmixed.legacy.oracles import LinearLMEOracleRegularized
from skmixed.legacy.solvers import LinearLMERegSolver

import numpy as np

supported_logger_keys = ('beta', 'gamma', 'tbeta', 'tgamma', 'loss', 'test_loss', 'converged')


class FeatureSelectorV1:

    def __init__(self, max_iter, tol):
        self.max_iter = max_iter
        self.tol = tol
        self.feature_selection_log_parameters = None
        self.num_features = None
        self.num_random_effects = None

    def fit(self,
            train: LinearLMEOracleRegularized,
            test: LinearLMEOracleRegularized,
            model: LinearLMERegSolver,
            beta0: np.ndarray,
            gamma0: np.ndarray,
            logger_keys=(),
            selection_mode = None,
            **kwargs):

        num_features = train.problem.num_features
        num_random_effects = train.problem.num_random_effects

        assert num_features == num_random_effects, "num_features must be equal to num_random_effects (it's only implemented for this case by now)"

        parameters = {
            (num_features + 1, num_random_effects): {"beta": beta0,
                                                     "gamma": gamma0}
        }

        # k is how much non-zero elements of beta we want to get
        for k in range(num_features, 0, -1):
            train.k = k
            test.k = k

            for j in range(k, 0, -1):
                train.j = j
                test.j = j

                # we initialize it taking the parameters from the previous level of sparsity
                if j == k:
                    prev_parameters = parameters[(k + 1, k)]
                else:
                    prev_parameters = parameters[(k, j + 1)]
                # just for testing purposes
                if selection_mode is "dummy":
                    prev_parameters = {"beta": beta0, "gamma": gamma0}

                prev_beta = prev_parameters["beta"]
                prev_gamma = prev_parameters["gamma"]

                tbeta = train.take_only_k_max(prev_beta, k)
                tgamma = prev_gamma
                tgamma[tbeta == 0] = 0
                tgamma = train.take_only_k_max(tgamma, j)
                prev_tbeta = np.infty
                prev_tgamma = np.infty
                logger = None
                inner_iters_total = 0
                for _ in range(self.max_iter):
                    # minimize w.r.t. beta/gamma
                    logger = model.fit(train,
                                       test,
                                       beta0=prev_beta,
                                       gamma0=prev_gamma,
                                       tbeta=tbeta,
                                       tgamma=tgamma,
                                       initializer=None,
                                       **kwargs
                                       )
                    # minimize w.r.t. tbeta/tgamma
                    tbeta = train.take_only_k_max(model.beta, k)
                    tgamma = model.gamma.copy()
                    tgamma[tbeta == 0] = 0
                    tgamma = train.take_only_k_max(tgamma, j)

                    inner_iters_total += 1
                    # stopping criterion is based on tbeta/tgamma
                    if np.linalg.norm(tbeta - prev_tbeta) < self.tol and \
                        np.linalg.norm(tgamma - prev_tgamma) < self.tol:
                            break
                    else:
                        prev_beta = model.beta
                        prev_gamma = model.gamma
                        prev_tbeta = tbeta
                        prev_tgamma = tgamma

                # Logging the result
                new_parameters = {
                    "beta": model.beta,
                    "gamma": model.gamma,
                    "tbeta": tbeta,
                    "tgamma": tgamma,
                    "logger": logger,
                    "inner_iters_total": inner_iters_total
                }

                if "converged" in logger_keys:
                    new_parameters["converged"] = logger["converged"]
                if "loss" in logger_keys:
                    loss = train.loss_reg(model.beta, model.gamma, model.tbeta, model.tgamma)
                    new_parameters["loss"] = loss
                if "test_loss" in logger_keys:
                    test_loss = test.loss_reg(model.beta, model.gamma, model.tbeta, model.tgamma)
                    new_parameters["test_loss"] = test_loss
                if "proj_grad_gamma_norm" in logger_keys:
                    grad_gamma = train.gradient_gamma_reg(model.beta, model.gamma, model.tgamma)
                    grad_gamma[model.gamma == 0] = 0  # now it's projected onto the constraints gamma >= 0
                    new_parameters["proj_grad_gamma_norm"] = np.linalg.norm(grad_gamma)
                if "loss_wo_reg" in logger_keys:
                    loss = train.loss(model.beta, model.gamma)
                    new_parameters["loss_wo_reg"] = loss
                if "loss_wo_reg_tgamma" in logger_keys:
                    loss = train.loss(tbeta, tgamma)
                    new_parameters["loss_wo_reg_tgamma"] = loss
                if "test_loss_wo_reg" in logger_keys:
                    loss = test.loss(model.beta, model.gamma)
                    new_parameters["test_loss_wo_reg"] = loss
                if "test_loss_wo_reg_tgamma" in logger_keys:
                    loss = test.loss(tbeta, tgamma)
                    new_parameters["test_loss_wo_reg_tgamma"] = loss


                parameters[(k, j)] = new_parameters
        self.feature_selection_log_parameters = parameters
        self.num_features = num_features
        self.num_random_effects = num_random_effects
        return 0

    def get_aggregated_parameter(self, key):
        assert self.feature_selection_log_parameters is not None, "You should launch .fit() first"
        if key in ('beta', 'gamma', 'tbeta', 'tgamma'):
            result = np.zeros((self.num_features, self.num_random_effects, self.num_features))
        else:
            result = np.zeros((self.num_features, self.num_random_effects))
        for k in range(self.num_features, 0, -1):
            for j in range(min(self.num_random_effects, k), 0, -1):
                parameters = self.feature_selection_log_parameters[(k, j)]
                if key in ('beta', 'gamma', 'tbeta', 'tgamma'):
                    result[k - 1, j - 1, :] = parameters[key]
                else:
                    result[k - 1, j - 1] = parameters[key]
        return result
