import numpy as np

from lib.oracles import LinearLMEOracle, LinearLMEOracleRegularized

supported_methods = ('EM', 'GradDescent', 'AccGradDescent', 'NewtonRaphson', 'VariableProjectionNR')
supported_logger_keys = ('loss', 'beta', 'gamma', 'test_loss', 'grad_gamma', 'hess_gamma')


class LinearLMESolver:
    def __init__(self, tol: float = 1e-4, max_iter: int = 1000, logger_keys=('loss', 'beta', 'gamma', 'grad_gamma')):
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

        assert not ("test_loss" in logger_keys and test is None), \
            "Logger keys contain test_loss, but the test oracle was not provided"

        self.logger = dict(zip(logger_keys, [[] for _ in logger_keys]))

    def log(self):
        if "loss" in self.logger.keys() :
            self.logger["loss"].append(self.loss)
        if "beta" in self.logger.keys():
            self.logger["beta"].append(self.beta)
        if "gamma" in self.logger.keys():
            self.logger["gamma"].append(self.gamma)
        if "test_loss" in self.logger.keys():
            self.logger['test_loss'].append(self.test_loss)
        if "grad_gamma" in self.logger.keys():
            self.logger['grad_gamma'].append(self.grad_gamma)
        if "hess_gamma" in self.logger.keys():
            self.logger['hess_gamma'].append(self.hess_gamma)

    def clean_copy(self):
        return LinearLMESolver(self.tol, self.max_iter)

    def fit(self, train: LinearLMEOracle, test: LinearLMEOracle = None, beta0=None, gamma0=None, method='EM'):

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

        gamma_step_length = (0.01 / i for i in range(1, self.max_iter))

        self.loss = train.loss(self.beta, self.gamma)
        self.grad_gamma = train.gradient_gamma(self.beta, self.gamma)
        self.log()
        # y_old = 0
        iteration = 0
        while np.linalg.norm(self.grad_gamma) > self.tol and iteration < self.max_iter:
            iteration += 1
            if iteration >= self.max_iter:
                self.us = train.optimal_random_effects(self.beta, self.gamma)
                self.logger["converged"] = 0
                return self.logger

            self.beta = train.optimal_beta(self.gamma)

            if method == 'GradDescent':
                # self.grad_gamma = train.gradient_gamma(self.beta, self.gamma)
                step_len = next(gamma_step_length)
                self.gamma = self.gamma - step_len * self.grad_gamma
            elif method == 'NewtonRaphson':
                self.hess_gamma = train.hessian_gamma(self.beta, self.gamma)
                # self.grad_gamma = train.gradient_gamma(self.beta, self.gamma)
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

            # TODO: Figure out how to ensure the positive constraint for gamma correctly
            self.gamma = np.clip(self.gamma, 0.01, None)

            self.grad_gamma = train.gradient_gamma(self.beta, self.gamma)
            self.loss = train.loss(self.beta, self.gamma)
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
            method='VariableProjectionNR', logger_keys=('loss', 'beta', 'gamma', 'hess_gamma')):

        assert method in supported_methods, \
            "Method %s is not from %s" % (method, sorted(supported_methods, key=lambda x: x[0]))

        assert logger_keys <= supported_logger_keys, \
            "Supported logger elements are: %s" % sorted(supported_logger_keys, key=lambda x: x[0])

        assert not ("test_loss" in logger_keys and test is None), \
            "Logger keys contain test_loss, but the test oracle was not provided"

        logger = dict(zip(logger_keys, [[] for _ in logger_keys]))

        k_gamma = train.problem.num_random_effects
        k_beta = train.problem.num_features

        if gamma0 is None:
            gamma = np.zeros(k_gamma) + 0.1
        else:
            assert len(gamma0) == k_gamma
            gamma = gamma0
        gamma = gamma.clip(0.1, None)

        if beta0 is None:
            beta = np.ones(k_beta)
        else:
            assert len(beta0) == k_beta
            beta = beta0

        tgamma = gamma
        tbeta = beta

        prev_loss = np.infty

        current_loss = train.loss_reg(beta, gamma, tbeta, tgamma)
        iteration = 0
        while abs(prev_loss - current_loss) > self.tol and iteration < self.max_iter:
            iteration += 1
            if iteration >= self.max_iter:
                self.beta = beta
                self.gamma = gamma
                self.tbeta = tbeta
                self.tgamma = tgamma
                self.us = train.optimal_random_effects(beta, gamma)
                logger["converged"] = 0
                return logger

            prev_loss = current_loss

            grad_gamma = None
            hess_gamma = None

            if method == 'VariableProjectionNR':
                beta = train.optimal_beta_reg(gamma, tbeta)
                tbeta = train.optimal_tbeta(beta)
                grad_gamma = train.gradient_gamma_reg(beta, gamma, tgamma)
                hess_gamma = train.hessian_gamma_reg(beta, gamma)
                gamma = gamma - np.linalg.solve(hess_gamma, grad_gamma)
                tgamma = train.optimal_tgamma(tbeta, gamma)

            # TODO: Figure out how to ensure the positive constraint for gamma correctly
            gamma = np.clip(gamma, 0, None)
            current_loss = train.loss_reg(beta, gamma, tbeta, tgamma)

            if "loss" in logger_keys:
                logger["loss"].append(current_loss)
            if "beta" in logger_keys:
                logger["beta"].append(beta)
            if "gamma" in logger_keys:
                logger["gamma"].append(gamma)
            if "test_loss" in logger_keys:
                logger['test_loss'].append(test.loss(beta, gamma))
            if "grad_gamma" in logger_keys:
                logger['grad_gamma'].append(grad_gamma)
            if "hess_gamma" in logger_keys:
                logger['hess_gamma'].append(hess_gamma)

        self.beta = beta
        self.gamma = gamma
        self.tbeta = tbeta
        self.tgamma = tgamma
        self.us = train.optimal_random_effects(beta, gamma)
        logger['converged'] = 1

        return logger


if __name__ == '__main__':
    def psd(hessian):
        eigvals = np.linalg.eigvals(hessian)
        if np.linalg.norm(np.imag(eigvals)) > 1e-15:
            return -1
        min_eigval = min(np.real(eigvals))
        if min_eigval < 0:
            return -1
        else:
            return min_eigval

    from lib.problems import LinearLMEProblem

    random_seed = 212
    study_sizes = [100, 50, 10]
    num_features = 6
    num_random_effects = 6
    obs_std = 5e-2
    method = "EM"
    z_same_as_x = True

    beta = np.ones(num_features)
    beta[-1] = 0
    gamma = np.ones(num_random_effects)
    gamma[-1] = 0.1

    train, beta, gamma, random_effects, errs = LinearLMEProblem.generate(study_sizes=study_sizes,
                                                                         num_features=num_features,
                                                                         beta=beta,
                                                                         num_random_effects=num_random_effects,
                                                                         gamma=gamma,
                                                                         z_same_as_x=z_same_as_x,
                                                                         obs_std=obs_std,
                                                                         seed=random_seed)

    empirical_gamma = np.sum(random_effects ** 2, axis=0) / len(study_sizes)

    test = LinearLMEProblem.generate(study_sizes=[10, 10, 10], beta=beta, gamma=gamma, z_same_as_x=z_same_as_x,
                                     true_random_effects=random_effects,
                                     seed=random_seed + 1, return_true_parameters=False)
    true_parameters = {
        "beta": beta,
        "gamma": gamma,
        "random_effects": random_effects,
        "errs": errs,
        "train": train,
        "test": test,
        "seed": random_seed
    }

    color_map = ["red", "green", "blue", "yellow", "black", "cyan", "purple", "orange"]

    lb=1
    lg1=1e-4
    lg2=1

    if method == "VariableProjectionNR":
        train_oracle = LinearLMEOracleRegularized(train, k=5, lb=lb, lg1=lg1, lg2=lg2)
        test_oracle = LinearLMEOracleRegularized(test, k=5, lb=lb, lg1=lg1, lg2=lg2)
        model = LinearLMERegSolver(tol=1e-8, max_iter=10)
    else:
        train_oracle = LinearLMEOracle(train, mode='naive')
        test_oracle = LinearLMEOracle(test, mode='naive')
        model = LinearLMESolver(tol=1e-8, max_iter=10)

    logger = model.fit(train_oracle, test_oracle,
                       beta0=np.ones(num_features),
                       gamma0=gamma*0.6 + 0.2,
                       method=method)