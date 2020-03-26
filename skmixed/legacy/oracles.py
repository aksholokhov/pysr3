import numpy as np
from scipy.optimize import minimize
from scipy.linalg.lapack import get_lapack_funcs

from skmixed.lme.problems import LinearLMEProblem


class LinearLMEOracle:
    def __init__(self, problem: LinearLMEProblem, mode='fast'):
        self.problem = problem
        assert mode == 'naive' or mode == 'fast', "Unknown mode: %s" % mode
        self.mode = mode
        if self.mode == 'fast':
            self.omegas_inv = []
            self.zTomegas_inv = []
            self.zTomegas_invZ = []
            self.xTomegas_invY = []
            self.xTomegas_invX = []
            self.old_gamma = None

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
                l_mat = np.diag(l)
                omega_inv = np.linalg.inv(z.dot(np.diag(gamma)).dot(z.T) + l_mat)
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
                dGamma_inv = np.diag(1 / (self.old_gamma - gamma))
                for i, (x, y, z, l) in enumerate(self.problem):
                    kernel_update = np.linalg.inv(dGamma_inv - self.zTomegas_invZ[i])
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

    def loss(self, beta: np.ndarray, gamma: np.ndarray) -> float:
        gamma_mat = np.diag(gamma)
        result = 0
        problem = self.problem
        if self.mode == 'naive':
            for x, y, z, l in problem:
                omega = z.dot(gamma_mat).dot(z.T) + np.diag(l)
                xi = y - x.dot(beta)
                sign, determinant = np.linalg.slogdet(omega)
                result += 1 / 2 * xi.T.dot(np.linalg.inv(omega)).dot(xi) + 1 / 2 * sign * determinant
        elif self.mode == 'fast':
            self.recalculate_inverse_matrices(gamma)
            for i, (x, y, z, l) in enumerate(problem):
                omega_inv = self.omegas_inv[i]
                xi = y - x.dot(beta)
                sign, determinant = np.linalg.slogdet(omega_inv)
                # Minus because we need the det of omega but use the det of its inverse
                result += 1 / 2 * xi.T.dot(omega_inv).dot(xi) - 1 / 2 * sign * determinant
        else:
            raise Exception("Unknown mode: %s" % self.mode)
        return result

    def gradient_gamma(self, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        if self.mode == 'naive':
            gamma_mat = np.diag(gamma)
            grad_gamma = np.zeros(len(gamma))
            for j in range(len(gamma)):
                result = 0
                for x, y, z, l in self.problem:
                    omega_inv = z.dot(gamma_mat).dot(z.T) + np.diag(l)
                    xi = y - x.dot(beta)
                    z_col = z[:, j]
                    data_part = z_col.T.dot(np.linalg.inv(omega_inv)).dot(xi)
                    data_part = -1 / 2 * data_part ** 2
                    det_part = 1 / 2 * z_col.T.dot(np.linalg.inv(omega_inv)).dot(z_col)
                    result += data_part + det_part
                grad_gamma[j] = result
            return grad_gamma
        elif self.mode == 'fast':
            if (self.old_gamma == gamma).all():
                result = np.zeros(self.problem.num_random_effects)
                for i, (x, y, z, l) in enumerate(self.problem):
                    xi = y - x.dot(beta)
                    result += 1 / 2 * (np.diag(self.zTomegas_invZ[i]) - self.zTomegas_inv[i].dot(xi) ** 2)
                self.old_gamma = gamma
                return result
            else:
                self.recalculate_inverse_matrices(gamma)
                result = np.zeros(self.problem.num_random_effects)
                for i, (x, y, z, l) in enumerate(self.problem):
                    new_zTomega = self.zTomegas_inv[i]
                    new_zTomegaZ = self.zTomegas_invZ[i]
                    xi = y - x.dot(beta)
                    result += 1 / 2 * (np.diag(new_zTomegaZ) - new_zTomega.dot(xi) ** 2)
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

    def optimal_beta(self, gamma: np.ndarray):  # , force_naive=False):
        omega = 0
        tail = 0

        if self.mode == 'naive':
            gamma_mat = np.diag(gamma)
            for x, y, z, l in self.problem:
                omega_i = z.dot(gamma_mat).dot(z.T) + np.diag(l)
                omega += x.T.dot(np.linalg.inv(omega_i)).dot(x)
                tail += x.T.dot(np.linalg.inv(omega_i)).dot(y)
        elif self.mode == 'fast':
            if not (self.old_gamma == gamma).all():
                self.recalculate_inverse_matrices(gamma)
            omega = np.sum(self.xTomegas_invX, axis=0)
            tail = np.sum(self.xTomegas_invY, axis=0)
        else:
            raise Exception("Unexpected mode: %s" % self.mode)
        return np.linalg.inv(omega).dot(tail)

    def optimal_random_effects(self, beta: np.ndarray, gamma: np.ndarray):
        random_effects = []
        for x, y, z, l in self.problem:
            l = np.diag(l)
            # This is an ad-hoc to make the matrix gamma invertible even when
            # some \gamma_i are zero. We fix it shortly after.
            # TODO: implement better account for zero gamma (need to del. resp. rows and columns from all the matrices)
            inv_g = np.diag(np.array([0 if g == 0 else 1 / g for g in gamma]))
            u = np.linalg.inv(inv_g + z.T.dot(np.linalg.inv(l)).dot(z)).dot(
                z.T.dot(np.linalg.inv(l)).dot(y - x.dot(beta)))
            # Here we put all the random effects for zero gammas to be zero
            for i, g in enumerate(gamma):
                if g == 0:
                    u[i] = 0
            random_effects.append(u)
        return np.array(random_effects)


class LinearLMEOracleRegularized(LinearLMEOracle):
    def __init__(self, problem: LinearLMEProblem, mode='fast', lb=0.1, lg=0.1, k=3, j=3):
        super().__init__(problem, mode)
        self.lb = lb
        self.lg = lg
        self.k = k
        self.j = j

    def optimal_beta_reg(self, gamma: np.ndarray, tbeta: np.ndarray):
        omega = 0
        tail = 0
        invert_upper_triangular = get_lapack_funcs("trtri")
        if self.mode == 'naive':
            gamma_mat = np.diag(gamma)
            for x, y, z, l in self.problem:
                omega_i = z.dot(gamma_mat).dot(z.T) + np.diag(l)
                L = np.linalg.cholesky(omega_i)
                #L_inv = invert_upper_triangular(L.T)[0].T
                #Lx = L_inv.dot(x)
                #omega += Lx.T.dot(Lx)
                omega += x.T.dot(np.linalg.inv(omega_i)).dot(x)
                #omega += np.diag(1/gamma)
                tail += x.T.dot(np.linalg.solve(omega_i, y))
        elif self.mode == 'fast':
            if not (self.old_gamma == gamma).all():
                self.recalculate_inverse_matrices(gamma)
            omega = np.sum(self.xTomegas_invX, axis=0)
            tail = np.sum(self.xTomegas_invY, axis=0)
        else:
            raise Exception("Unexpected mode: %s" % self.mode)
        return np.linalg.solve(self.lb * np.eye(self.problem.num_fixed_effects) + omega, self.lb * tbeta + tail)

    @staticmethod
    def take_only_k_max(a: np.ndarray, k: int):
        b = np.zeros(len(a))
        idx_k_max = a.argsort()[-k:]
        b[idx_k_max] = a[idx_k_max]
        return b

    def optimal_tbeta(self, beta: np.ndarray):
        return self.take_only_k_max(beta, self.k)

    def loss_reg(self, beta: np.ndarray, gamma: np.ndarray, tbeta: np.ndarray, tgamma: np.ndarray):
        return self.loss(beta, gamma) + self.lb / 2 * sum((beta - tbeta) ** 2) + \
               self.lg / 2 * sum((gamma - tgamma) ** 2)

    def gradient_gamma_reg(self, beta: np.ndarray, gamma: np.ndarray, tgamma: np.ndarray) -> np.ndarray:
        return self.gradient_gamma(beta, gamma) + self.lg * (gamma - tgamma)

    def hessian_gamma_reg(self, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        return self.hessian_gamma(beta, gamma) + self.lg * np.eye(self.problem.num_random_effects)

    def optimal_tgamma(self, tbeta, gamma):
        tgamma = np.zeros(len(gamma))
        idx = tbeta != 0
        tgamma[idx] = gamma[idx]
        return self.take_only_k_max(tgamma, self.j)

    def good_lambda_gamma(self, mode="upperbound"):
        if mode == "upperbound":
            return sum([np.linalg.norm(self.problem.random_features[i]) ** 4 / np.max(
                self.problem.obs_stds[i] ** 2)
                        for i in range(self.problem.num_groups)])

        elif mode == "exact":
            def minus_max_eig(gamma: np.ndarray):
                self.recalculate_inverse_matrices(gamma)
                A = np.sum([s * s for s in self.zTomegas_invZ], axis=0)
                return -max(np.linalg.eigvals(A))

            gamma_opt = minimize(minus_max_eig, np.ones(self.problem.num_random_effects),
                                 bounds=[(0, None)] * self.problem.num_random_effects).x
            return -minus_max_eig(gamma_opt)

        elif mode == "exact_full_hess":
            def minus_max_eig(gamma: np.ndarray):
                tbeta = np.zeros(self.problem.num_fixed_effects)
                beta = self.optimal_beta_reg(gamma, tbeta)
                return np.min(np.linalg.eigvals(self.hessian_gamma(beta, gamma)))

            gamma_opt = minimize(minus_max_eig, np.ones(self.problem.num_random_effects),
                                 bounds=[(0, 5)] * self.problem.num_random_effects).x
            return -minus_max_eig(gamma_opt), gamma_opt

        else:
            raise Exception("Unknown mode: %s" % mode)
