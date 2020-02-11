from lib.oracles import LinearLMEOracleRegularized
from lib.solvers import LinearLMERegSolver

import numpy as np

supported_logger_keys = ('beta', 'gamma', 'tbeta', 'tgamma', 'loss', 'test_loss', 'converged')


class FeatureSelectorV1:

    def __init__(self, max_iter):
        self.max_ter = max_iter
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
            # we initialize it taking the parameters from the previous level of sparsity
            prev_parameters = parameters[(k + 1, k)]
            prev_beta = prev_parameters["beta"]
            tbeta = train.take_only_k_max(prev_beta, k)

            for j in range(k, 0, -1):
                train.j = j
                test.j = j

                if j == k:
                    prev_parameters = parameters[(k + 1, k)]
                else:
                    prev_parameters = parameters[(k, j + 1)]

                prev_beta = prev_parameters["beta"]
                prev_gamma = prev_parameters["gamma"]

                tgamma = prev_gamma
                tgamma[tbeta == 0] = 0
                tgamma = train.take_only_k_max(tgamma, j)
                #print(k, j, ": \n Beta: ", prev_beta, '\n Gamma: ', prev_gamma, '\n tbeta: ', tbeta, '\n tgamma:',
                #      tgamma, '\n')
                logger = model.fit(train,
                                   test,
                                   beta0=prev_beta,
                                   gamma0=prev_gamma,
                                   tbeta=tbeta,
                                   tgamma=tgamma,
                                   initializer=None,
                                   **kwargs
                                   )
                #print(k, j, " after: \n Beta: ", model.beta, '\n Gamma: ', model.gamma, '\n tbeta: ', model.tbeta,
                 #     '\n tgamma:', model.tgamma, '\n')

                # Logging the result
                new_parameters = {
                    "beta": model.beta,
                    "gamma": model.gamma,
                    "tbeta": model.tbeta,
                    "tgamma": model.tgamma,
                    "logger": logger
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
