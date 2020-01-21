import numpy as np

from lib.oracles import LinearLMEOracle, LinearLMEOracleRegularized

supported_methods = ('EM', 'GradDescent', 'AccGradDescent', 'NewtonRaphson')
supported_logger_keys = ('loss', 'beta', 'gamma', 'test_loss', 'grad_gamma', 'hess_gamma')

class LinearLMESolver:
    def __init__(self, tol: float = 1e-4, max_iter: int = 1000):
        self.tol = tol
        self.max_iter = max_iter
        self.beta = None
        self.gamma = None
        self.us = None

    def clean_copy(self):
        return LinearLMESolver(self.tol, self.max_iter)

    def fit(self, train: LinearLMEOracle, test: LinearLMEOracle = None, beta0=None, gamma0=None, method='EM',
            logger_keys=('loss', 'beta', 'gamma')):

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
            gamma =  np.zeros(k_gamma) + 0.1
        else:
            assert len(gamma0) == k_gamma
            gamma = gamma0

        if beta0 is None:
            beta = np.ones(k_beta)
        else:
            assert len(beta0) == k_beta
            beta = beta0

        gamma_step_length = (0.01 / i for i in range(1, self.max_iter))

        prev_loss = np.infty

        current_loss = train.loss(beta, gamma)
        y_old = 0
        iteration = 0
        while abs(prev_loss - current_loss) > self.tol and iteration < self.max_iter:
            iteration += 1
            if iteration >= self.max_iter:
                self.beta = beta
                self.gamma = gamma
                self.us = train.optimal_random_effects(beta, gamma)
                logger["converged"] = 0
                return logger

            prev_loss = current_loss
            beta = train.optimal_beta(gamma)

            grad_gamma = None
            hess_gamma = None

            if method == 'GradDescent':
                grad_gamma = train.gradient_gamma(beta, gamma)
                step_len = next(gamma_step_length)
                gamma = gamma - step_len * grad_gamma
            elif method == 'NewtonRaphson':
                hess_gamma = train.hessian_gamma(beta, gamma)
                grad_gamma = train.gradient_gamma(beta, gamma)
                gamma = gamma - np.linalg.solve(hess_gamma, grad_gamma)
            elif method == 'AccGradDescent':
                grad_gamma = train.gradient_gamma(beta, gamma)
                step_len = next(gamma_step_length)
                y_new = gamma - step_len * grad_gamma
                gamma = y_new + 0.1 * (y_new - y_old)
                y_old = y_new
            elif method == 'EM':
                us = train.optimal_random_effects(beta, gamma)
                # Naive vay of calculating gamma:
                gamma = np.sum(us ** 2, axis=0) / train.problem.num_studies

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

            # TODO: Figure out how to ensure the positive constraint for gamma correctly
            gamma = np.clip(gamma, 0.01, None)
            current_loss = train.loss(beta, gamma)

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
        self.us = train.optimal_random_effects(beta, gamma)
        logger['converged'] = 1

        return logger

    def __str__(self):
        output = "LinearLMESolver \n"
        output += " Beta: " + ' '.join([str(t) for t in self.beta]) + '\n'
        output += " Gamma: " + ' '.join([str(t) for t in self.gamma]) + '\n'
        output += " Random effects: \n "
        output += '\n '.join([str(t) for t in self.us])
        return output


class LinearLMERegSolver(LinearLMESolver):
    def fit(self, train: LinearLMEOracleRegularized, test: LinearLMEOracleRegularized = None, beta0=None, gamma0=None, method='EM',
            logger_keys=('loss', 'beta', 'gamma')):

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
                tgamma = train.optimal_tgamma(gamma)

            # TODO: Figure out how to ensure the positive constraint for gamma correctly
            gamma = np.clip(gamma, 0.01, None)
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


