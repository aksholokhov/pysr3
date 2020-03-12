import numpy as np

from skmixed.legacy.oracles import LinearLMEOracle, LinearLMEOracleRegularized

supported_methods = ('EM', 'GradDescent', 'AccGradDescent', 'NewtonRaphson', 'VariableProjectionNR', "VariableProjectionGD")
supported_logger_keys = ('loss', 'beta', 'gamma', 'test_loss', 'grad_gamma', 'hess_gamma')


class LinearLMESolver:
    def __init__(self, tol: float = 1e-4, max_iter: int = 1000,
                 logger_keys=('loss', 'beta', 'gamma', 'grad_gamma', 'grad_gamma_norm', 'test_loss')):
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
            proj_grad = self.grad_gamma.copy()
            proj_grad[self.gamma == 0] = 0
            self.logger['grad_gamma_norm'].append(np.linalg.norm(proj_grad))
        if "hess_gamma" in self.logger.keys():
            self.logger['hess_gamma'].append(self.hess_gamma)

    def clean_copy(self):
        return LinearLMESolver(self.tol, self.max_iter)

    def fit(self, train: LinearLMEOracle, test: LinearLMEOracle = None, initializer="None", beta0=None, gamma0=None,
            method='EM', **kwargs):
        """Generates something

        :param train:
        :param test:
        :param initializer:
        :param beta0:
        :param gamma0:
        :param method:
        :param kwargs:
        :return:
        """
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

    def fit(self,
                train: LinearLMEOracleRegularized,
                test: LinearLMEOracleRegularized = None,
                beta0=None,
                gamma0=None,
                method='VariableProjectionGD',
                initializer="None",
                tbeta: np.ndarray = None,
                tgamma: np.ndarray = None,
                use_line_search=True,
                init_step_len=1, **kwargs):


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

        # initializing beta and gamma with one naive iteration of EM algorithm
        if initializer == "EM":
            self.beta = train.optimal_beta(self.gamma)
            self.us = train.optimal_random_effects(self.beta, self.gamma)
            self.gamma = np.sum(self.us ** 2, axis=0) / train.problem.num_studies
            # self.tgamma = self.gamma

        self.loss = train.loss_reg(self.beta, self.gamma, self.tbeta, self.tgamma)
        self.grad_gamma = train.gradient_gamma_reg(self.beta, self.gamma, self.tgamma)
        projected_gradient = self.grad_gamma.copy()
        self.log()

        iteration = 0
        while np.linalg.norm(projected_gradient) > self.tol and iteration < self.max_iter:
            iteration += 1
            if iteration >= self.max_iter:
                self.us = train.optimal_random_effects(self.beta, self.gamma)
                self.logger["converged"] = 0
                return self.logger

            if method == 'VariableProjectionGD':
                self.beta = train.optimal_beta_reg(self.gamma, self.tbeta)
                self.grad_gamma = train.gradient_gamma_reg(self.beta, self.gamma, self.tgamma)
                # self.hess_gamma = train.hessian_gamma_reg(self.beta, self.gamma)
                # direction = -np.linalg.solve(self.hess_gamma, self.grad_gamma)
                # direction = -np.linalg.solve(self.hess_gamma.T.dot(self.hess_gamma), self.hess_gamma.T.dot(self.grad_gamma))
                #direction = -self.grad_gamma
                projected_gradient = self.grad_gamma.copy()
                projected_gradient[self.gamma == 0] = 0
                direction = -projected_gradient
                if use_line_search:
                    # line search method
                    step_len = init_step_len  # min(1, np.max(-self.gamma/direction))
                    current_loss = train.loss_reg(self.beta, self.gamma, self.tbeta, self.tgamma)
                    while train.loss_reg(self.beta, self.gamma + step_len * direction, self.tbeta,
                                         self.tgamma) >= current_loss:
                        step_len *= 0.5
                        if step_len <= 1e-9:
                            break
                else:
                    # fixed step size
                    step_len = 1 / iteration
                self.gamma = self.gamma + step_len*direction

            # projection

            self.gamma = np.clip(self.gamma, 0, None)

            self.grad_gamma = train.gradient_gamma_reg(self.beta, self.gamma, self.tgamma)
            projected_gradient = self.grad_gamma.copy()
            projected_gradient[self.gamma == 0] = 0
            self.loss = train.loss_reg(self.beta, self.gamma, self.tbeta, self.tgamma)
            if "test_loss" in self.logger.keys():
                self.test_loss = test.loss_reg(self.beta, self.gamma, self.tbeta, self.tgamma)
            self.log()

        self.us = train.optimal_random_effects(self.beta, self.gamma)
        self.logger['converged'] = 1
        self.logger['iterations'] = iteration

        return self.logger


if __name__ == '__main__':

    from linear_mixed_effects.problems import LinearLMEProblem

    random_seed = 212
    study_sizes = [300, 100, 50]
    test_study_sizes = [10, 10, 10]
    num_features = 6
    num_random_effects = 6
    obs_std = 5e-2
    method = "VariableProjectionGD"
    initializer = "EM"
    lb = 1
    lg = 1
    how_close = 1
    tol = 1e-4
    max_iter = 100

    beta = np.ones(num_features)
    gamma = np.ones(num_random_effects)

    # beta is [0, 1, 1, 1 ...]
    # gamma is [0, 1, 1, ..., 1, 0]
    beta[0] = 0
    gamma[0] = 0
    gamma[-1] = 0

    train, beta, gamma, random_effects, errs = LinearLMEProblem.generate(groups_sizes=study_sizes,
                                                                         num_fixed_effects=num_features,
                                                                         beta=beta,
                                                                         gamma=gamma,
                                                                         num_random_effects=num_random_effects,
                                                                         how_close_z_to_x=how_close,
                                                                         obs_std=obs_std,
                                                                         seed=random_seed)

    empirical_gamma = np.sum(random_effects ** 2, axis=0) / len(study_sizes)

    test = LinearLMEProblem.generate(groups_sizes=test_study_sizes, beta=beta, gamma=gamma,
                                     how_close_z_to_x=how_close,
                                     true_random_effects=random_effects,
                                     seed=random_seed + 1,
                                     obs_std=obs_std,
                                     return_true_model_coefficients=False)
    true_parameters = {
        "beta": beta,
        "gamma": gamma,
        "random_effects": random_effects,
        "errs": errs,
        "train": train,
        "test": test,
        "seed": random_seed
    }

    if method == "VariableProjectionGD":
        # these are oracle and method which are capable of performing feature selection
        train_oracle = LinearLMEOracleRegularized(train, lb=lb, lg=lg)
        test_oracle = LinearLMEOracleRegularized(test, lb=lb, lg=lg)
        model = LinearLMERegSolver(tol=tol, max_iter=max_iter)
        # this says what regularizer we need to put in order to make the hessian PSD everywhere
        # UPD: we don't use it, because we don't use NR anymore (we use GD instead)
        # gamma_reg_exact_full, g_opt = train_oracle.good_lambda_gamma(mode="exact_full_hess")
        # print(r"Suggested $\lambda_\gamma$ adjustment: ", gamma_reg_exact_full)
        # print(r"Used $\lambda_\gamma$: ", lg)
    else:
        # these methods are left here for experimenting, they don't work for feature selection
        # because they behave badly near the boundaries
        train_oracle = LinearLMEOracle(train)
        test_oracle = LinearLMEOracle(test)
        model = LinearLMESolver(tol=tol, max_iter=max_iter)

    # We start the feature selection process from a "one EM iteration" initial point
    model.max_iter = 0
    logger = model.fit(train_oracle,
                       test_oracle,
                       beta0=np.ones(num_features),
                       gamma0=np.ones(num_random_effects),
                       initializer="EM",
                       )
    model.max_iter = max_iter

    # These are initial parameters for the very first iteration
    parameters = {
        (num_features + 1, num_random_effects): (model.beta, model.gamma, None)
    }

    # k is how much non-zero elements of beta we want to get
    for k in range(num_features, 0, -1):
        train_oracle.k = k
        test_oracle.k = k
        prev_beta, *rest = parameters[(k + 1, k)]
        tbeta = train_oracle.take_only_k_max(prev_beta, k)

        for j in range(k, 0, -1):
            train_oracle.j = j
            test_oracle.j = j
            if j == k:
                prev_beta, prev_gamma, *rest = parameters[(k + 1, k)]
            else:
                prev_beta, prev_gamma, *rest = parameters[(k, j + 1)]

            tgamma = prev_gamma
            tgamma[tbeta == 0] = 0
            tgamma = train_oracle.take_only_k_max(tgamma, j)

            logger = model.fit(train_oracle,
                               test_oracle,
                               beta0=prev_beta,
                               gamma0=prev_gamma,
                               tbeta=tbeta,
                               tgamma=tgamma,
                               method=method,
                               initializer=None,
                               use_line_search=True
                               )

            train_loss = train_oracle.loss_reg(model.beta, model.gamma, tbeta, tgamma)
            test_loss = test_oracle.loss_reg(model.beta, model.gamma, tbeta, tgamma)
            if not logger["converged"]:
                parameters[(k, j)] = (prev_beta, prev_gamma, tbeta, tgamma, logger, train_loss, test_loss)
            else:
                parameters[(k, j)] = (model.beta, model.gamma, model.tbeta, model.tgamma, logger, train_loss, test_loss)

    train_error = np.zeros((num_features, num_features))
    test_error = np.zeros((num_features, num_features))
    converged = np.zeros((num_features, num_features))
    coefficients = np.zeros((num_features * num_features, 2 * num_features))
    dense_coefficients = np.zeros((num_features * num_features, 2 * num_features))
    for k in range(num_features, 0, -1):
        for j in range(k, 0, -1):
            beta, gamma, tbeta, tgamma, logger, train_loss, test_loss = parameters[(k, j)]
            train_error[k - 1, j - 1] = train_loss
            test_error[k - 1, j - 1] = test_loss
            converged[k - 1, j - 1] = logger["converged"]

            dense_coefficients[(k - 1) * num_features:k * num_features, 2 * (j - 1)] = beta
            dense_coefficients[(k - 1) * num_features:k * num_features, 2 * (j - 1) + 1] = gamma

            if k == num_features:
                coefficients[(k - 1) * num_features:k * num_features, 2 * (j - 1)] = beta
                coefficients[(k - 1) * num_features:k * num_features, 2 * (j - 1) + 1] = tgamma
            else:
                coefficients[(k - 1) * num_features:k * num_features, 2 * (j - 1)] = tbeta
                coefficients[(k - 1) * num_features:k * num_features, 2 * (j - 1) + 1] = tgamma

    coefficients[:num_features, 2 * (num_features - 1)] = true_parameters["beta"]
    coefficients[:num_features, 2 * (num_features - 1) + 1] = empirical_gamma

    dense_coefficients[:num_features, 2 * (num_features - 1)] = true_parameters["beta"]
    dense_coefficients[:num_features, 2 * (num_features - 1) + 1] = empirical_gamma
