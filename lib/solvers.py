import numpy as np
from matplotlib import pyplot as plt

from lib.oracles import LinearLMEOracle, LinearLMEOracleRegularized

supported_methods = ('EM', 'GradDescent', 'AccGradDescent', 'NewtonRaphson', 'VariableProjectionNR')
supported_logger_keys = ('loss', 'beta', 'gamma', 'test_loss', 'grad_gamma', 'hess_gamma')


class LinearLMESolver:
    def __init__(self, tol: float = 1e-4, max_iter: int = 1000, logger_keys=('loss', 'beta', 'gamma', 'grad_gamma', 'grad_gamma_norm', 'test_loss')):
        self.tol = tol
        self.max_iter = max_iter
        self.loss = None
        self.test_loss = None
        self.beta = None
        self.gamma = None
        self.us = None
        self.grad_gamma = None
        self.hess_gamma = None
        self.method = None

        assert logger_keys <= supported_logger_keys, \
            "Supported logger elements are: %s" % sorted(supported_logger_keys, key=lambda x: x[0])

        self.logger = dict(zip(logger_keys, [[] for _ in logger_keys]))

    def log(self):
        if "loss" in self.logger.keys():
            self.logger["loss"].append(self.loss)
        if "beta" in self.logger.keys():
            self.logger["beta"].append(self.beta)
        if "gamma" in self.logger.keys():
            self.logger["gamma"].append(self.gamma)
        if "test_loss" in self.logger.keys():
            self.logger['test_loss'].append(self.test_loss)
        if "grad_gamma" in self.logger.keys():
            self.logger['grad_gamma'].append(self.grad_gamma)
        if "grad_gamma_norm" in self.logger.keys():
            self.logger['grad_gamma_norm'].append(np.linalg.norm(self.grad_gamma))
        if "hess_gamma" in self.logger.keys():
            self.logger['hess_gamma'].append(self.hess_gamma)

    def clean_copy(self):
        return LinearLMESolver(self.tol, self.max_iter)

    def fit(self, train: LinearLMEOracle, test: LinearLMEOracle = None, initializer="None", beta0=None, gamma0=None, method='EM', **kwargs):

        assert method in supported_methods, \
            "Method %s is not from %s" % (method, sorted(supported_methods, key=lambda x: x[0]))
        self.method = method

        k_gamma = train.problem.num_random_effects
        k_beta = train.problem.num_features

        if gamma0 is None:
            self.gamma = np.zeros(k_gamma) + 0.1
        else:
            assert len(gamma0) == k_gamma
            self.gamma = gamma0

        if beta0 is None:
            self.beta = np.ones(k_beta)
        else:
            assert len(beta0) == k_beta
            self.beta = beta0

        if initializer == "EM":
            self.beta = train.optimal_beta(self.gamma)
            self.us = train.optimal_random_effects(self.beta, self.gamma)
            self.gamma = np.sum(self.us ** 2, axis=0) / train.problem.num_studies

        gamma_step_length = (1 / i for i in range(1, self.max_iter))

        self.loss = train.loss(self.beta, self.gamma)
        self.grad_gamma = train.gradient_gamma(self.beta, self.gamma)
        self.log()
        # y_old = 0
        iteration = 0
        prev_gamma = np.infty
        while np.linalg.norm(self.grad_gamma) > self.tol and iteration < self.max_iter:
            iteration += 1
            if iteration >= self.max_iter:
                self.us = train.optimal_random_effects(self.beta, self.gamma)
                self.logger["converged"] = 0
                return self.logger

            prev_gamma = self.gamma

            self.beta = train.optimal_beta(self.gamma)
            if method == 'GradDescent':
                self.grad_gamma = train.gradient_gamma(self.beta, self.gamma)
                step_len = next(gamma_step_length)
                self.gamma = self.gamma - step_len * self.grad_gamma
            elif method == 'NewtonRaphson':
                self.hess_gamma = train.hessian_gamma(self.beta, self.gamma)
                self.grad_gamma = train.gradient_gamma(self.beta, self.gamma)
                self.gamma = self.gamma - np.linalg.solve(self.hess_gamma, self.grad_gamma)
            # elif method == 'AccGradDescent':
            #     # self.grad_gamma = train.gradient_gamma(self.beta, self.gamma)
            #     y_new = self.gamma - step_len * self.grad_gamma
            #     self.gamma = y_new + 0.1 * (y_new - y_old)
            #     y_old = y_new
            elif method == 'EM':
                self.us = train.optimal_random_effects(self.beta, self.gamma)
                # Naive vay of calculating gamma:
                self.gamma = np.sum(self.us ** 2, axis=0) / train.problem.num_studies

                # # Actual EM algorithm from papers
                # new_gamma = 0
                #
                # def var_u(i) -> np.ndarray:
                #     gamma_col = gamma.reshape((len(gamma), 1))
                #     core = inv(np.sum(self.xTomegas_invX, axis=0))
                #     GG = gamma_col.dot(gamma_col.T)
                #     zOx = self.zTomegas_inv[i].dot(self.problem.features[i])
                #     res = GG*self.zTomegas_invZ[i] - GG*(zOx.dot(core).dot(zOx.T))
                #     return res
                #
                # for i, u in enumerate(us):
                #     u = u.reshape(len(u), 1)
                #     new_gamma += u.dot(u.T) + var_u(i)
                # gamma = np.diag(new_gamma)/problem.num_studies
            else:
                raise Exception("Unknown Method")

            self.gamma = np.clip(self.gamma, 0, None)

            self.grad_gamma = train.gradient_gamma(self.beta, self.gamma)
            self.loss = train.loss(self.beta, self.gamma)
            if "test_loss" in self.logger.keys():
                self.test_loss = test.loss(self.beta, self.gamma)
            self.log()

        self.us = train.optimal_random_effects(self.beta, self.gamma)
        self.logger['converged'] = 1

        return self.logger

    def __str__(self):
        output = "LinearLMESolver \n"
        output += " Beta: " + ' '.join([str(t) for t in self.beta]) + '\n'
        output += " Gamma: " + ' '.join([str(t) for t in self.gamma]) + '\n'
        output += " Random effects: \n "
        output += '\n '.join([str(t) for t in self.us])
        return output


class LinearLMERegSolver(LinearLMESolver):

    def fit(self, train: LinearLMEOracleRegularized, test: LinearLMEOracleRegularized = None, beta0=None, gamma0=None,
            method='VariableProjectionNR', initializer="None", tbeta: np.ndarray = None, tgamma: np.ndarray = None, **kwargs):

        assert method in supported_methods, \
            "Method %s is not from %s" % (method, sorted(supported_methods, key=lambda x: x[0]))

        self.method = method

        k_gamma = train.problem.num_random_effects
        k_beta = train.problem.num_features

        if gamma0 is None:
            self.gamma = np.zeros(k_gamma) + 0.1
        else:
            assert len(gamma0) == k_gamma
            self.gamma = gamma0

        if beta0 is None:
            self.beta = np.ones(k_beta)
        else:
            assert len(beta0) == k_beta
            self.beta = beta0

        if tgamma is None:
            self.tgamma = np.zeros(k_gamma)
        else:
            assert len(tgamma) == k_gamma
            self.tgamma = tgamma

        if tbeta is None:
            self.tbeta = np.zeros(k_beta)
        else:
            assert len(tbeta) == k_beta
            self.tbeta = tbeta


        if initializer == "EM":
            self.beta = train.optimal_beta(self.gamma)
            self.us = train.optimal_random_effects(self.beta, self.gamma)
            self.gamma = np.sum(self.us ** 2, axis=0) / train.problem.num_studies
            #self.tgamma = self.gamma


        self.loss = train.loss_reg(self.beta, self.gamma, self.tbeta, self.tgamma)
        self.grad_gamma = train.gradient_gamma_reg(self.beta, self.gamma, self.tgamma)
        self.log()

        prev_gamma = np.infty
        iteration = 0
        while np.linalg.norm(self.gamma - prev_gamma) > self.tol and iteration < self.max_iter:
            iteration += 1
            if iteration >= self.max_iter:
                self.us = train.optimal_random_effects(self.beta, self.gamma)
                self.logger["converged"] = 0
                return self.logger

            prev_gamma = self.gamma
            if method == 'VariableProjectionNR':
                self.beta = train.optimal_beta_reg(self.gamma, self.tbeta)
                self.grad_gamma = train.gradient_gamma_reg(self.beta, self.gamma, self.tgamma)
                # self.hess_gamma = train.hessian_gamma_reg(self.beta, self.gamma)
                #direction = -np.linalg.solve(self.hess_gamma, self.grad_gamma)
                #direction = -np.linalg.solve(self.hess_gamma.T.dot(self.hess_gamma), self.hess_gamma.T.dot(self.grad_gamma))
                direction = -1/iteration*self.grad_gamma
                self.gamma = self.gamma + direction

            # projection

            self.gamma = np.clip(self.gamma, 0, None)

            self.grad_gamma = train.gradient_gamma_reg(self.beta, self.gamma, self.tgamma)
            self.loss = train.loss_reg(self.beta, self.gamma, self.tbeta, self.tgamma)
            if "test_loss" in self.logger.keys():
                self.test_loss = test.loss_reg(self.beta, self.gamma, self.tbeta, self.tgamma)
            self.log()

        self.us = train.optimal_random_effects(self.beta, self.gamma)
        self.logger['converged'] = 1

        return self.logger


if __name__ == '__main__':

    from lib.problems import LinearLMEProblem

    random_seed = 212
    study_sizes = [300, 100, 50]
    test_study_sizes = [10, 10, 10]
    num_features = 10
    num_random_effects = 10
    obs_std = 5e-2
    method = "NewtonRaphson"
    initializer = "EM"
    lb = 1
    how_close = 1
    tol = 1e-4

    beta = np.ones(num_features)
    #beta[-2:] = 0
    gamma = np.ones(num_random_effects)
    gamma[-2:] = 0
    train, beta, gamma, random_effects, errs = LinearLMEProblem.generate(study_sizes=study_sizes,
                                                                         num_features=num_features,
                                                                         beta=beta,
                                                                         gamma=gamma,
                                                                         num_random_effects=num_random_effects,
                                                                         how_close_z_to_x=how_close,
                                                                         obs_std=obs_std,
                                                                         seed=random_seed)

    empirical_gamma = np.sum(random_effects ** 2, axis=0) / len(study_sizes)

    test = LinearLMEProblem.generate(study_sizes=test_study_sizes, beta=beta, gamma=gamma,
                                     how_close_z_to_x=how_close,
                                     true_random_effects=random_effects,
                                     seed=random_seed + 1,
                                     obs_std=obs_std,
                                     return_true_parameters=False)
    true_parameters = {
        "beta": beta,
        "gamma": gamma,
        "random_effects": random_effects,
        "errs": errs,
        "train": train,
        "test": test,
        "seed": random_seed
    }

    if method == "VariableProjectionNR":
        train_oracle = LinearLMEOracleRegularized(train, lb=0, lg=0, k=num_features, j=num_random_effects)
        test_oracle = LinearLMEOracleRegularized(test, lb=0, lg=0, k=num_features, j=num_random_effects)
        #gamma_reg_exact_full, g_opt = train_oracle.good_lambda_gamma(mode="exact_full_hess")
        gamma_reg_exact_full = 1e3
        train_oracle.lg = gamma_reg_exact_full
        test_oracle.lg = gamma_reg_exact_full
        model = LinearLMERegSolver(tol=tol, max_iter=1000)
    else:
        train_oracle = LinearLMEOracle(train)
        test_oracle = LinearLMEOracle(test)
        model = LinearLMESolver(tol=tol, max_iter=1000)

    tbeta = np.zeros(num_features)
    tgamma = np.ones(num_random_effects)

    logger = model.fit(train_oracle, test_oracle, method=method, initializer=initializer, tbeta=tbeta, tgamma=tgamma)

    parameters = {
        (num_features + 1, num_random_effects): (np.ones(num_features), np.zeros(num_random_effects) + 0.1, None)
    }

    for k in range(num_features, 0, -1):
        train_oracle.k = k
        test_oracle.k = k
        if k == num_features:
            train_oracle.lb = 0
            test_oracle.lb = 0
            tbeta = np.zeros(num_features)
        else:
            train_oracle.lb = lb
            test_oracle.lb = lb
            prev_beta, *rest = parameters[(k + 1, k)]
            tbeta = train_oracle.take_only_k_max(prev_beta, k)
        for j in range(k, 0, -1):
            train_oracle.j = j
            test_oracle.j = j
            if j == k:
                prev_beta, prev_gamma, *rest = parameters[(k + 1, j)]
            else:
                prev_beta, prev_gamma, *rest = parameters[(k, j + 1)]
            tgamma = train_oracle.take_only_k_max(prev_gamma, j)

            logger = model.fit(train_oracle,
                               test_oracle,
                               beta0=prev_beta,
                               gamma0=prev_gamma,
                               tbeta=tbeta,
                               tgamma=tgamma
                               )
            train_loss = train_oracle.loss_reg(model.beta, model.gamma, tbeta, tgamma)
            test_loss = test_oracle.loss_reg(model.beta, model.gamma, tbeta, tgamma)
            if not logger["converged"]:
                parameters[(k, j)] = (prev_beta, prev_gamma, tbeta, tgamma, logger, train_loss, test_loss)
            else:
                parameters[(k, j)] = (model.beta, model.gamma, model.tbeta, model.tgamma, logger, train_loss, test_loss)





