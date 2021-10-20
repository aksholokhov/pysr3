import numpy as np
import scipy as sp

from pysr3.linear.problems import LinearProblem
from pysr3.priors import Prior, NonInformativePrior
from pysr3.glms.link_functions import LinkFunction, IdentityLinkFunction


class GLMOracle:

    def __init__(self, problem: LinearProblem = None, prior: Prior = None, link_function: LinkFunction = None):
        self.problem = problem
        self.prior = prior if prior else NonInformativePrior()
        self.link_function = link_function if link_function else IdentityLinkFunction

    def instantiate(self, problem):
        self.problem = problem
        self.prior.instantiate(problem)

    def forget(self):
        self.problem = None
        self.prior.forget()

    def loss(self, x):
        a = self.problem.a
        b = self.problem.b
        return (1 / self.problem.obs_std ** 2 * (
                self.link_function.value(a.dot(x)) - b * a.dot(x))).sum() + self.prior.loss(x)

    def gradient(self, x):
        a = self.problem.a
        b = self.problem.b
        return (a.T * (1 / self.problem.obs_std ** 2 * (
                self.link_function.gradient(a.dot(x)) - b))).T.sum(axis=0) + self.prior.gradient(x)

    def hessian(self, x):
        res = 0
        for i, ai in enumerate(self.problem.a):
            ai = ai.reshape(-1, 1)
            res = res + (1 / self.problem.obs_std[i] ** 2) * self.link_function.hessian(ai.T.dot(x)) * ai.dot(ai.T)
        return res + self.prior.hessian(x)

    def value_function(self, x):
        return self.loss(x)

    def gradient_value_function(self, x):
        return self.gradient(x)

    def aic(self, x):
        p = sum(x != 0)
        return self.loss(x) + 2 * p

    def bic(self, x):
        p = sum(x != 0)
        return self.loss(x) + np.log(self.problem.num_objects) * p


class GLMOracleSR3(GLMOracle):

    def __init__(self, problem: LinearProblem = None, lam=1, practical=False, prior: Prior = None,
                 link_function: LinkFunction = None, constraints=None, do_correction_steps=True):
        assert not prior, "Priors for GLMOracleSR3 are not supported yet"
        super(GLMOracleSR3, self).__init__(problem=problem, prior=prior, link_function=link_function)
        self.constraints = constraints
        self.lam = lam
        self.practical = practical

    def instantiate(self, problem):
        self.problem = problem
        if self.constraints is None:
            self.constraints = ([-np.infty]*problem.num_features, [np.infty]*problem.num_features)
        self.prior.instantiate(problem)

    def forget(self):
        self.problem = None
        self.prior.forget()

    def loss(self, x, w=None):
        a = self.problem.a
        b = self.problem.b
        return (super().loss(x) +
                + self.lam / 2 * np.linalg.norm(x - w, ord=2) ** 2
                + self.prior.loss(x))

    def gradient_x(self, x, w):
        return super().gradient(x) + self.lam * (x - w)

    def value_function(self, w, **kwargs):
        x = self.find_optimal_parameters(w, **kwargs)
        return self.loss(x, w)

    def gradient_value_function(self, w, **kwargs):
        x = self.find_optimal_parameters(w, **kwargs)
        return -self.lam * (x - w)

    def find_optimal_parameters(self, w, x0=None, regularizer=None, tol=1e-4, max_iter=1000, mu_decay=0.5, logger=None,
                                do_correction_steps=False, **kwargs):
        x = x0 if x0 is not None else np.ones(len(w)) / 3
        step_len = 1 / self.lam
        x_prev = np.infty
        iteration = 0

        # generate constraints index set and matrix
        constraint_index_set = []
        a = []
        b = []
        idx_constr_a = []
        idx_constr_x = []
        i = 0
        j = 0
        for l, r in zip(*self.constraints):
            a_local = []
            if -np.infty < l:
                a_local.append([-1])
                b.append(-l)
                idx_constr_a.append(i)
                idx_constr_x.append(j)
            else:
                a_local.append([0])
                b.append(0)
            i += 1
            if r < np.infty:
                a_local.append([1])
                b.append(r)
                idx_constr_a.append(i)
                idx_constr_x.append(j)
            else:
                a_local.append([0])
                b.append(0)
            i += 1
            j += 1
            a.append(np.array(a_local))

        a = sp.linalg.block_diag(*a)
        have_constraints = (a != 0).any()
        if have_constraints:
            idx_constr_a = np.array(idx_constr_a)
            idx_constr_x = np.array(idx_constr_x)
            n = len(idx_constr_a)
            b = np.array(b)
            a_nnz = a[idx_constr_a, :]
            b_nnz = b[idx_constr_a]
            v = np.ones(n)/5
            mu = np.nan_to_num(mu_decay * v.dot(x[idx_constr_x]) / n, 0)

            def G(v, x, mu):
                return np.concatenate([
                    v * (b_nnz - a_nnz.dot(x)) - mu,
                    self.gradient_x(x, w) + (a_nnz * v[:, np.newaxis]).sum(axis=0)
                ])

            def dG(v, x):
                return np.block([
                    [np.diag(b_nnz - a_nnz.dot(x)), -a_nnz * v[:, np.newaxis]],
                    [a_nnz.T, self.hessian(x) + self.lam * np.eye(len(x))]
                ])

        else:
            def G(v, x, mu):
                return self.gradient_x(x, w)

            def dG(v, x):
                return self.hessian(x) + self.lam * np.eye(len(x))

            v = None
            mu = None

        is_correction_step = False

        while np.linalg.norm(x - x_prev) > tol \
                and iteration < max_iter:
            x_prev = x
            # get new direction
            G_current = G(v, x, mu)
            dG_current = dG(v, x)
            direction = np.linalg.solve(dG_current, -G_current)
            if have_constraints:
                # establish the maximum length of step_size based on v+ >= 0
                dv = direction[:len(v)]
                dx = direction[len(v):]
                ind_neg_dir = np.where(dv < 0.0)[0]
                max_step_len = min(1, 1 if len(ind_neg_dir) == 0 else np.min(-v[ind_neg_dir] / dv[ind_neg_dir]))
                # establish the maximum length of step_size based on Ax+ <= b
                step_len = max_step_len
                while any(a.dot(x + step_len*dx) > b):
                    step_len *= 0.9
                v = v + step_len * dv
                x = x + step_len * dx
                if do_correction_steps and not is_correction_step:
                    mu = np.nan_to_num(v.dot(x[idx_constr_x]) / n, 0)
                    is_correction_step = True
                else:
                    mu = np.nan_to_num(mu_decay * v.dot(x[idx_constr_x]) / n, 0)
                    if do_correction_steps:
                        is_correction_step = False
            else:
                dx = direction
                x = x + dx

            iteration += 1

            if logger and len(logger.keys) > 0:
                logger.log(locals())
        if logger:
            logger.add("iteration", iteration)

        return x

    def aic(self, x):
        p = sum(x != 0)
        return super(self).loss(x) + 2 * p

    def bic(self, x):
        p = sum(x != 0)
        return super(self).loss(x) + np.log(self.problem.num_objects) * p