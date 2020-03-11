import numpy as np
from numpy import diag
from numpy.linalg import inv

from problems import LinearLMEProblem


class LinearLMESolver:
    def __init__(self, tol: float = 1e-4, max_iter: int = 1000, mode='naive', method='gd'):
        self.problem = None
        self.test_problem = None
        self.beta = None
        self.gamma = None
        self.us = None
        self.tol = tol
        self.max_iter = max_iter
        assert mode == 'naive' or mode == 'fast', "Unknown mode: %s" % mode
        self.mode = mode
        self.method = method
        # for fast gamma gradients TODO: fix variables naming
        if self.mode == 'fast':
            self.omegas_inv = []
            self.zTomegas_inv = []
            self.zTomegas_invZ = []
            self.xTomegas_invY = []
            self.xTomegas_invX = []
            self.old_gamma = None

    def clean_copy(self):
        return LinearLMESolver(self.tol, self.max_iter, self.mode, self.method)

    def recalculate_inverse_matrices(self, gamma: np.ndarray) -> None:
        if (self.old_gamma == gamma).all():
            return None
        if self.old_gamma is None:
            self.omegas_inv = []
            self.zTomegas_inv = []
            self.zTomegas_invZ = []
            self.xTomegas_invY = []
            self.xTomegas_invX = []
            for x, y, z, l in self.problem:
                omega_inv = inv(z.dot(diag(gamma)).dot(z.T) + l)
                zTomega = z.T.dot(omega_inv)
                zTomegaZ = zTomega.dot(z)
                self.omegas_inv.append(omega_inv)
                self.zTomegas_inv.append(zTomega)
                self.zTomegas_invZ.append(zTomegaZ)
                xTomega_inv = x.T.dot(omega_inv)
                self.xTomegas_invX.append(xTomega_inv.dot(x))
                self.xTomegas_invY.append(xTomega_inv.dot(y))
            self.old_gamma = gamma
        else:
            if (self.old_gamma - gamma == 0).any():
                self.old_gamma = None
                self.recalculate_inverse_matrices(gamma)
            else:
                dGamma_inv = diag(1 / (self.old_gamma - gamma))
                for i, (x, y, z, l) in enumerate(self.problem):
                    kernel_update = inv(dGamma_inv - self.zTomegas_invZ[i])
                    new_omega = self.omegas_inv[i] + self.omegas_inv[i].dot(z).dot(kernel_update).dot(
                        self.zTomegas_inv[i])
                    new_zTomega = z.T.dot(new_omega)
                    new_zTomegaZ = new_zTomega.dot(z)
                    self.omegas_inv[i] = new_omega
                    self.zTomegas_inv[i] = new_zTomega
                    self.zTomegas_invZ[i] = new_zTomegaZ
                    xTomega_inv = x.T.dot(new_omega)
                    self.xTomegas_invY[i] = xTomega_inv.dot(y)
                    self.xTomegas_invX[i] = xTomega_inv.dot(x)
                self.old_gamma = gamma

    def loss(self, beta, gamma, use_test=False) -> float:
        gamma_mat = diag(gamma)
        result = 0
        if use_test:
            if self.test_problem is None:
                raise Exception("Test dataset has not been provided")
            problem = self.test_problem
        else:
            problem = self.problem
        # TODO: implement fast mode for test data
        if self.mode == 'naive' or use_test:
            for x, y, z, l in problem:
                omega = z.dot(gamma_mat).dot(z.T) + l
                xi = y - x.dot(beta)
                sign, determinant = np.linalg.slogdet(omega)
                result += 1 / 2 * xi.T.dot(inv(omega)).dot(xi) + 1 / 2 * sign * determinant
        elif self.mode == 'fast':
            self.recalculate_inverse_matrices(gamma)
            for i, (x, y, z, l) in enumerate(problem):
                omega_inv = self.omegas_inv[i]
                xi = y - x.dot(beta)
                # TODO: FIX THIS SHIT EVERYWHERE
                sign, determinant = np.linalg.slogdet(omega_inv)
                result += 1 / 2 * xi.T.dot(omega_inv).dot(xi) - 1 / 2 * sign * determinant
        else:
            raise Exception("Unknown mode: %s" % self.mode)
        return result

    def rml_loss(self, beta, gamma, use_test=False) -> float:
        gamma_mat = diag(gamma)
        result = 0
        if use_test:
            if self.test_problem is None:
                raise Exception("Test dataset has not been provided")
            problem = self.test_problem
        else:
            problem = self.problem
        # TODO: implement fast mode for test data
        if self.mode == 'naive' or use_test:
            for x, y, z, l in problem:
                omega = z.dot(gamma_mat).dot(z.T) + l
                xi = y - x.dot(beta)
                sign, determinant = np.linalg.slogdet(omega)
                sign2, determinant2 = np.linalg.slogdet(x.T.dot(inv(omega)).dot(x))
                result += 1 / 2 * xi.T.dot(inv(omega)).dot(
                    xi) + 1 / 2 * sign * determinant + 1 / 2 * sign2 * determinant2
        elif self.mode == 'fast':
            self.recalculate_inverse_matrices(gamma)
            for i, (x, y, z, l) in enumerate(problem):
                omega_inv = self.omegas_inv[i]
                sign, determinant = np.linalg.slogdet(omega_inv)
                sign2, determinant2 = np.linalg.slogdet(self.xTomegas_invX[i])
                xi = y - x.dot(beta)
                result += 1 / 2 * xi.T.dot(omega_inv).dot(
                    xi) - 1 / 2 * sign * determinant + 1 / 2 * sign2 * determinant2
        else:
            raise Exception("Unknown mode: %s" % self.mode)
        return result

    def grad_loss_gamma(self, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        if self.mode == 'naive':
            gamma_mat = diag(gamma)
            grad_gamma = np.zeros(len(gamma))
            for j in range(len(gamma)):
                result = 0
                for x, y, z, l in self.problem:
                    omega_inv = z.dot(gamma_mat).dot(z.T) + l
                    xi = y - x.dot(beta)
                    z_col = z[:, j]
                    data_part = z_col.T.dot(inv(omega_inv)).dot(xi)
                    data_part = -1 / 2 * data_part ** 2
                    det_part = 1 / 2 * z_col.T.dot(inv(omega_inv)).dot(z_col)
                    result += data_part + det_part
                grad_gamma[j] = result
            return grad_gamma
        elif self.mode == 'fast':
            if (self.old_gamma == gamma).all():
                result = np.zeros(self.problem.num_random_effects)
                for i, (x, y, z, l) in enumerate(self.problem):
                    xi = y - x.dot(beta)
                    result += 1 / 2 * (diag(self.zTomegas_invZ[i]) - self.zTomegas_inv[i].dot(xi) ** 2)
                self.old_gamma = gamma
                return result
            else:
                self.recalculate_inverse_matrices(gamma)
                result = np.zeros(self.problem.num_random_effects)
                for i, (x, y, z, l) in enumerate(self.problem):
                    new_zTomega = self.zTomegas_inv[i]
                    new_zTomegaZ = self.zTomegas_invZ[i]
                    xi = y - x.dot(beta)
                    result += 1 / 2 * (diag(new_zTomegaZ) - new_zTomega.dot(xi) ** 2)
                return result

    def hessian_gamma(self, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        if self.mode == 'naive':
            raise NotImplementedError(
                "Hessians are not implemented for the naive mode (it takes forever to compute them)")
        elif self.mode == 'fast':
            result = np.zeros((self.problem.num_random_effects, self.problem.num_random_effects))
            self.recalculate_inverse_matrices(gamma)
            for i, (x, y, z, l) in enumerate(self.problem):
                xi = y - x.dot(beta)
                eta = self.zTomegas_inv[i].dot(xi)
                eta = eta.reshape(len(eta), 1)
                result -= self.zTomegas_invZ[i] ** 2
                result += 2 * eta.dot(eta.T) * self.zTomegas_invZ[i]
            return 1 / 2 * result
        else:
            raise Exception("Unknown mode: %s" % self.mode)

    def hessian_criterion(self, beta, gamma):
        result = np.zeros((self.problem.num_random_effects, self.problem.num_random_effects))
        self.recalculate_inverse_matrices(gamma)
        for i, (x, y, z, l) in enumerate(self.problem):
            xi = y - x.dot(beta)
            xi = xi.reshape(len(xi), 1)
            result += self.zTomegas_invZ[i]*z.T.dot(
                2 * self.omegas_inv[i].dot(xi.dot(xi.T)).dot(self.omegas_inv[i]) - self.omegas_inv[i]
            ).dot(z)
        return 1/2*result

    def optimal_beta(self, gamma: np.ndarray, force_naive=False):
        omega = 0
        tail = 0
        if self.mode == 'naive' or force_naive:
            gamma_mat = diag(gamma)
            for x, y, z, l in self.problem:
                omega_i = z.dot(gamma_mat).dot(z.T) + l
                omega += x.T.dot(inv(omega_i)).dot(x)
                tail += x.T.dot(inv(omega_i)).dot(y)
        elif self.mode == 'fast':
            if (self.old_gamma == gamma).all():
                omega = np.sum(self.xTomegas_invX, axis=0)
                tail = np.sum(self.xTomegas_invY, axis=0)
            else:
                return self.optimal_beta(gamma, force_naive=True)
        else:
            raise Exception("Unexpected mode: %s" % self.mode)
        return inv(omega).dot(tail)

    def optimal_random_effects(self, beta, gamma):
        random_effects = []
        for x, y, z, l in self.problem:
            inv_g = np.diag(np.array([0 if g == 0 else 1 / g for g in gamma]))
            u = inv(inv_g + z.T.dot(inv(l)).dot(z)).dot(z.T.dot(inv(l)).dot(y - x.dot(beta)))
            random_effects.append(u)
        return np.array(random_effects)

    def fit(self, problem: LinearLMEProblem, test: LinearLMEProblem = None, gamma0=None, beta0=None,
            no_calculations=False, track_intermediate_loss=False):
        if problem.answers is None:
            raise Exception("You need to provide answers for training")

        self.old_gamma = None
        self.problem = problem
        if test is not None:
            self.test_problem = test

        k_gamma = problem.num_random_effects
        k_beta = problem.num_fixed_effects

        if gamma0 is None:
            gamma = np.ones(k_gamma)
        else:
            assert len(gamma0) == k_gamma
            gamma = gamma0

        if beta0 is None:
            beta = np.ones(k_beta)
        else:
            assert len(beta0) == k_beta
            beta = beta0

        # recreate computationally hard parts once
        if self.mode == 'fast':
            self.recalculate_inverse_matrices(gamma)

        if no_calculations:
            return None

        logger = {
            "loss": [],
            "beta": [],
            "gamma": [],
            "first_em_gamma": [],
            "test_loss": [],
            "intermediate_loss": [],
            "hess": []
        }
        if self.method == 'gd':
            gamma_step_length = (0.1 / i for i in range(1, self.max_iter))
        elif self.method == 'nr':
            gamma_step_length = (1 / i for i in range(1, self.max_iter))
        else:
            gamma_step_length = (0.1 / i for i in range(1, self.max_iter))

        prev_loss = np.infty

        current_loss = self.loss(beta, gamma)
        y_old = 0
        first_iter = True
        while abs(prev_loss - current_loss) > self.tol:
            prev_loss = current_loss
            beta = self.optimal_beta(gamma)

            try:
                step_len = next(gamma_step_length)
            except StopIteration:
                logger["converged"] = 0
                return logger
            if track_intermediate_loss:
                logger["loss"].append(self.loss(beta, gamma))
                if self.test_problem is not None:
                    logger['test_loss'].append(self.loss(beta, gamma, use_test=True))
            if self.method == 'gd':
                # hess = self.hessian_gamma(beta, gamma)
                # logger["hess"].append(hess)
                gamma = gamma - step_len * self.grad_loss_gamma(beta, gamma)
            elif self.method == 'nr':
                # us = self.optimal_random_effects(beta, gamma)
                # gamma = np.sum(us ** 2, axis=0) / problem.num_studies
                # if first_iter:
                #     logger["first_em_gamma"].append(gamma)
                #     first_iter = False
                hess = self.hessian_gamma(beta, gamma)
                grad = self.grad_loss_gamma(beta, gamma)
                gamma = gamma - np.linalg.solve(hess, grad)
            elif self.method == 'agd':
                grad = self.grad_loss_gamma(beta, gamma)
                y_new = gamma - step_len * grad
                gamma = y_new + 0.1 * (y_new - y_old)
                y_old = y_new
            elif self.method == 'em':
                us = self.optimal_random_effects(beta, gamma)
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

                # Naive vay of calculating gamma
                # Also, it's just wrong because we know that mean is zero,
                # but np.var does not take it into account
                # gamma = np.var(us, axis=0)

                # Correct naive vay of doing that:
                gamma = np.sum(us ** 2, axis=0) / (problem.num_studies)

            gamma = np.clip(gamma, 0.01, None)
            self.recalculate_inverse_matrices(gamma)
            current_loss = self.loss(beta, gamma)
            logger["loss"].append(current_loss)
            logger["beta"].append(beta)
            logger["gamma"].append(gamma)
            if self.test_problem is not None:
                logger['test_loss'].append(self.loss(beta, gamma, use_test=True))

        self.beta = beta
        self.gamma = gamma
        self.us = self.optimal_random_effects(beta, gamma)
        logger['converged'] = 1

        return logger

    def predict(self, problem: LinearLMEProblem):
        if self.beta is None or self.gamma is None or self.us is None:
            raise Exception("The model has not been trained yet. Call .fit() for your train data")
        answers = []
        for i, (x, _, z, l) in enumerate(problem):
            y = x.dot(self.beta) + z.dot(self.us[i])
            answers.append(y)
        return answers

    def __str__(self):
        output = "LinearLMESolver \n"
        output += " Beta: " + ' '.join([str(t) for t in self.beta]) + '\n'
        output += " Gamma: " + ' '.join([str(t) for t in self.gamma]) + '\n'
        output += " Random effects: \n "
        output += '\n '.join([str(t) for t in self.us])
        return output


if __name__ == "__main__":
    noise_variance = 1e-2
    loss_tol = 1e-5
    random_seed = 32

    problem, beta, gamma, us, ls = LinearLMEProblem.generate(groups_sizes=[100, 100, 100],
                                                             num_fixed_effects=8,
                                                             num_random_effects=2, obs_std=noise_variance,
                                                             seed=random_seed, return_true_model_coefficients=True)
    alg1 = LinearLMESolver(tol=loss_tol, mode='fast', method='gd')
    logger_gd = alg1.fit(problem, no_calculations=True)
    gamma0 = np.ones(problem.num_random_effects)
    beta0 = alg1.optimal_beta(gamma0)
    hess0 = alg1.hessian_criterion(beta, gamma0)
    pass
