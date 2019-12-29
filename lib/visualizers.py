from math import ceil

import numpy as np
from dask.distributed import Client, as_completed
from matplotlib import pyplot as plt

from lib.bootstrappers import NonParLinearLMEBootstrapper, LMEBootstrapper
from lib.problems import LinearLMEProblem
from lib.solvers import LinearLMESolver


# from deprecated import deprecated

class LMEModelVisualizer:

    def __init__(self):
        self.color_map = ["red", "green", "blue", "yellow", "black", "cyan", "purple", "orange"]  # TODO: make better color generator
        self.model = None
        self.logger = None
        self.train = None
        self.test = None
        self.train_predictions = None
        self.test_predictions = None
        self.true_parameters = None
        self.bootstrap_models = []
        self.bootstrap_train_predictions = []
        self.bootstrap_test_predictions = []
        self.bootstrap_loggers = []
        self.min_z = None
        pass

    def fit(self, model: LinearLMESolver, train: LinearLMEProblem, test: LinearLMEProblem = None, true_parameters=None,
            bootstrapper: LMEBootstrapper = None, progressbar=None, client=None, ) -> None:

        def fit_one(assignment):
            result = []
            for local_model, local_train, local_test, full_train in assignment:
                logger = local_model.fit(local_train, test=local_test)
                pred_train = local_model.predict(full_train) if logger["converged"] else None
                pred_test = local_model.predict(local_test) if logger["converged"] else None
                result.append((local_model, logger, pred_train, pred_test))
            return result

        tasks_per_worker = 10
        num_bootstraps = 100
        if client is None:
            client = Client(processes=False)
        main_model = model.clean_copy()
        main_model_trainer = client.submit(fit_one, [(main_model, train, test, train)])
        if bootstrapper is not None:
            # assignments = []
            piece_of_work = tasks_per_worker / num_bootstraps
            futures = []
            bootstaps_counter = 0
            for _ in range(ceil(num_bootstraps / tasks_per_worker)):
                assignment = []
                for i, bootstrap_train in zip(
                        range(bootstaps_counter, min(num_bootstraps, bootstaps_counter + tasks_per_worker), 1),
                        bootstrapper):
                    local_model = model.clean_copy()
                    assignment.append((local_model, bootstrap_train, test, train))
                    bootstaps_counter += 1
                # assignments.append(assignment)
                futures.append(client.submit(fit_one, assignment))
            # data_futures = client.scatter(assignments)
            # for data_future in data_futures:
            #    futures.append(client.submit(fit_one, data_future))
            for thread in as_completed(futures):
                for local_model, logger, pred_train, pred_test in thread.result():
                    if logger['converged'] == 0:
                        continue
                    self.bootstrap_models.append(local_model)
                    self.bootstrap_loggers.append(logger)
                    self.bootstrap_train_predictions.append(pred_train)
                    if test is not None:
                        self.bootstrap_test_predictions.append(pred_test)
                if progressbar is not None:
                    progressbar.value += piece_of_work
                else:
                    print('piece done')
        main_model, main_logger, pred_train, pred_test = main_model_trainer.result()[0]
        client.close()
        if progressbar is not None:
            progressbar.value = 1
        self.model = main_model
        self.logger = main_logger
        self.train = train
        self.train_predictions = pred_train
        if test is not None:
            self.test = test
            self.test_predictions = pred_test
        if true_parameters is not None:
            self.true_parameters = true_parameters
        return main_logger['converged']

    def __predictions_plot(self, ax, real_answers, predictions, bootstrap_predictions=None, with_confidences=False):
        for i, (y_true, y_pred) in enumerate(zip(real_answers, predictions)):
            ax.scatter(y_true, y_pred, label="Study %d: %d train objects %s" % (
                i + 1, self.train.study_sizes[i],
                "" if self.test is None else ", %d test objects" % self.test.study_sizes[i]),
                       c=self.color_map[i])
            ax.legend()
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c=".3")
        ax.set_xlabel("True answers")
        ax.set_ylabel("Predictions")
        if with_confidences:
            if len(self.bootstrap_models) == 0:
                raise Warning("No bootstraper were provided, so no confidence intervals will be displayed")
            else:
                bootstrap_predictions = np.array(bootstrap_predictions)

                for i, (y_true, y_pred) in enumerate(zip(real_answers, predictions)):
                    predictions_low_err = np.abs(np.percentile(bootstrap_predictions[:, i, :], q=5, axis=0) - y_pred)
                    predictions_high_err = np.abs(np.percentile(bootstrap_predictions[:, i, :], q=95, axis=0) - y_pred)
                    ax.errorbar(y_true, y_pred, xerr=np.array([predictions_low_err, predictions_high_err]),
                                fmt='none', c=self.color_map[i])

    def plot_test_predictions(self, ax, with_confidences=True):
        return self.__predictions_plot(ax, self.test.answers, self.test_predictions, self.bootstrap_test_predictions,
                                       with_confidences)

    def plot_parameters(self, ax, with_confidences=True):
        pred_beta = self.model.beta
        ax.scatter(self.true_parameters['beta'], pred_beta)
        ax.set_xlabel("True parameters")
        ax.set_ylabel("Inferred parameters")
        model_parameters_low_lim = min(min(pred_beta), min(self.true_parameters['beta'])) - 0.1
        model_parameters_high_lim = max(max(pred_beta), max(self.true_parameters['beta'])) + 0.1
        ax.set_xlim(model_parameters_low_lim, model_parameters_high_lim)
        ax.set_ylim(model_parameters_low_lim, model_parameters_high_lim)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c=".3")
        if with_confidences:
            bootstrap_parameters = [model.beta for model in self.bootstrap_models]
            parameters_low_err = np.abs(np.percentile(np.array(bootstrap_parameters), q=5, axis=0) - pred_beta)
            parameters_high_err = np.abs(np.percentile(np.array(bootstrap_parameters), q=95, axis=0) - pred_beta)
            ax.errorbar(self.true_parameters['beta'], pred_beta,
                        xerr=np.array([parameters_low_err, parameters_high_err]), fmt='none')

    def plot_random_effects(self, ax, with_confidences=True):
        random_effects_low_lim = 0
        random_effects_high_lim = 0
        for i, (u_pred, u_true) in enumerate(zip(self.model.us, self.true_parameters['random_effects'])):
            ax.scatter(u_true, u_pred, label="Study %d" % (i + 1), c=self.color_map[i])
            random_effects_low_lim = min(min(u_true), min(u_pred), random_effects_low_lim)
            random_effects_high_lim = max(max(u_true), max(u_pred), random_effects_high_lim)
        ax.legend()
        random_effects_low_lim -= 0.1
        random_effects_high_lim += 0.1
        ax.set_xlim(random_effects_low_lim, random_effects_high_lim)
        ax.set_ylim(random_effects_low_lim, random_effects_high_lim)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c=".3")
        ax.set_xlabel("True random effects")
        ax.set_ylabel("Inferred random effects")
        if with_confidences:
            bootstrap_random_parameters = [model.us for model in self.bootstrap_models]
            bootstrap_random_parameters = np.array(bootstrap_random_parameters)
            for i, (u_pred, u_true) in enumerate(zip(self.model.us, self.true_parameters['random_effects'])):
                random_parameters_low_err = np.abs(
                    np.percentile(bootstrap_random_parameters[:, i, :], q=5, axis=0) - u_pred)
                random_parameters_high_err = np.abs(
                    np.percentile(bootstrap_random_parameters[:, i, :], q=95, axis=0) - u_pred)
                ax.errorbar(u_true, u_pred,
                            xerr=np.array([random_parameters_low_err, random_parameters_high_err]),
                            c=self.color_map[i],
                            fmt='none')

    def plot_gammas(self, ax, with_confidences=True):
        ax.scatter(self.true_parameters['gamma'], self.model.gamma, label="Inferred")
        #ax.scatter(self.true_parameters['gamma'], np.sqrt(np.var(self.true_parameters["random_effects"], axis=0)),
        #           label="avg. true")
        gamma_low_lim = min(min(self.true_parameters['gamma']), min(self.model.gamma)) - 0.2
        gamma_high_lim = max(max(self.true_parameters['gamma']), max(self.model.gamma)) + 0.2
        ax.set_xlim(gamma_low_lim, gamma_high_lim)
        ax.set_ylim(gamma_low_lim, gamma_high_lim)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c=".3")
        ax.set_xlabel("True gamma")
        ax.set_ylabel("Inferred gamma")
        if with_confidences:
            bootstrap_gamma = [model.gamma for model in self.bootstrap_models]
            pred_gamma = self.model.gamma
            gamma_low_err = np.abs(np.percentile(np.array(bootstrap_gamma), q=5, axis=0) - pred_gamma)
            gamma_high_err = np.abs(np.percentile(np.array(bootstrap_gamma), q=95, axis=0) - pred_gamma)
            ax.errorbar(self.true_parameters['gamma'], pred_gamma, xerr=np.array([gamma_low_err, gamma_high_err]),
                        fmt='none')

    def plot_loss(self, ax, loss_scale='log'):
        if self.min_z is not None:
            min_loss = min(self.min_z, np.min(self.logger["test_loss"]))
        else:
            min_loss = min(np.min(self.logger["loss"]), np.min(self.logger["test_loss"]))
        min_loss -= 1e-16
        if loss_scale == "log":
            loss_normed = np.array(self.logger["loss"]) - min_loss
            ax.semilogy(loss_normed, label="Train loss")
        else:
            ax.plot(self.logger["loss"], label="Train loss")

        if self.test is not None:
            if loss_scale == "log":
                test_loss_normed = np.array(self.logger["test_loss"]) - min_loss
                ax.semilogy(test_loss_normed, label="Test loss")
            else:
                ax.plot(self.logger["test_loss"], label="Test loss")
        ax.legend()

    def plot_gamma_trace(self, ax, loss_rml=False):
        assert len(self.model.gamma) == 2
        gamma0 = np.ones(2)
        gamma_trace = np.array(self.logger["gamma"]).T
        cmap = np.linspace(0.1, 0.9, len(gamma_trace[0]))
        ax.scatter(gamma_trace[0], gamma_trace[1], c=cmap, label=self.model.method)
        ax.scatter(gamma0[0], gamma0[1], c='g', label='start point')
        true_gamma = self.true_parameters["gamma"]
        ax.scatter(true_gamma[0], true_gamma[1], c='r', label='true gamma')
        true_random_effects = self.true_parameters["random_effects"]
        empirical_gamma = np.sum(true_random_effects ** 2, axis=0) / self.train.num_studies
        ax.scatter(empirical_gamma[0], empirical_gamma[1], c='pink', label='empirical gamma')
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        if not loss_rml:
            eps = 1
        else:
            eps = 1
        x = np.linspace(0, xlims[1] + eps, 100)
        y = np.linspace(0, ylims[1] + eps, 100)
        beta = self.logger['beta'][-1]
        # z = np.array([[self.model.loss(self.model.beta, np.array([g1, g2])) for g1 in x] for g2 in y])
        #prev_mode = self.model.mode
        #self.model.mode = 'naive'
        if loss_rml:
            z = np.array([[self.model.rml_loss(beta, np.array([g1, g2])) for g1 in x] for g2 in y])
        else:

            z = np.array([[self.model.loss(beta, np.array([g1, g2])) for g1 in x] for g2 in y])
        #self.model.mode = prev_mode
        # z = np.log10(z - self.min_z + 1e-16)

        levels = np.min(z) + np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
        cs = ax.contour(x, y, z, levels=levels)
        plt.clabel(cs, fontsize=8)
        ax.legend()
        if not loss_rml:
            self.min_z = np.min(z)

    def plot_hessian(self, ax, sufficient_criterion=False):
        assert len(self.model.gamma) == 2
        gamma0 = np.ones(2)
        gamma_trace = np.array(self.logger["gamma"]).T
        cmap = np.linspace(0.1, 0.9, len(gamma_trace[0]))
        ax.scatter(gamma_trace[0], gamma_trace[1], c=cmap, label=self.model.method)
        ax.scatter(gamma0[0], gamma0[1], c='g', label='start point')
        true_gamma = self.true_parameters["gamma"]
        ax.scatter(true_gamma[0], true_gamma[1], c='r', label='true gamma')
        true_random_effects = self.true_parameters["random_effects"]
        empirical_gamma = np.sum(true_random_effects ** 2, axis=0) / self.train.num_studies
        ax.scatter(empirical_gamma[0], empirical_gamma[1], c='pink', label='empirical gamma')
        #if self.model.method is "nr" or "em":
        #    ax.scatter(self.logger["first_em_gamma"][0][0], self.logger["first_em_gamma"][0][1], c='orange', label='first EM gamma')
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        eps = 1
        beta = self.logger['beta'][-1]
        plot_resolution = 100
        x = np.linspace(0, xlims[1] + eps, plot_resolution)
        y = np.linspace(0, ylims[1] + eps, plot_resolution)
        z = np.zeros((plot_resolution, plot_resolution))
        def psd(hessian):
            eigvals = np.linalg.eigvals(hessian)
            if np.linalg.norm(np.imag(eigvals)) > 1e-15:
                return -1
            min_eigval = min(np.real(eigvals))
            if min_eigval < 0:
                return -1
            else:
                return min_eigval

        for i, g2 in enumerate(y):
            for j, g1 in enumerate(x):
                gamma0 = np.array([g1, g2])
                beta0 = self.model.optimal_beta(gamma0)
                if sufficient_criterion:
                    hessian = self.model.hessian_criterion(beta0, gamma0)
                else:
                    hessian = self.model.hessian_gamma(beta0, gamma0)
                z[i, j] = psd(hessian)

        levels = [0, 1, 2]
        cs = ax.contour(x, y, z, levels=levels)
        plt.clabel(cs, fontsize=8)
        ax.legend()

if __name__ == '__main__':
    random_seed = 42
    train, beta, gamma, random_effects, errs = LinearLMEProblem.generate(study_sizes=[130, 25, 5],
                                                                         num_features=6,
                                                                         num_random_effects=3,
                                                                         obs_std=0.1,
                                                                         seed=random_seed)
    test = LinearLMEProblem.generate(study_sizes=[5, 5, 5], beta=beta, gamma=gamma,
                                     true_random_effects=random_effects,
                                     seed=random_seed + 1, return_true_parameters=False)
    true_parameters = {
        "beta": beta,
        "gamma": gamma,
        "random_effects": random_effects
    }

    model = LinearLMESolver(mode='fast')
    bootstrapper = NonParLinearLMEBootstrapper(train, max_capacity=100, seed=random_seed)
    visualizer = LMEModelVisualizer()
    visualizer.fit(model, train, test, true_parameters, bootstrapper)
    pass
