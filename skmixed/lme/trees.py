import numpy as np

from skmixed.lme.models import LinearLMESparseModel
from skmixed.lme.problems import LinearLMEProblem


class Tree:
    def __init__(self, model: LinearLMESparseModel, max_depth = 3, information_criterion="IC_vaida2005aic", minimal_gain = 1e1):
        self.model = model
        self.max_depth = max_depth
        self.information_criterion = information_criterion
        self.minimal_gain = minimal_gain
        self.coef_ = None

    def fit(self, problem):
        chosen_features = [0]
        possible_features = np.arange(1, problem.num_categorical_features)
        problem = problem.pivot((0, ))
        solver = self.model.copy()
        solver.logger_keys = solver.logger_keys | (self.information_criterion, )
        solver.fit(problem)
        previous_information_gain = solver.logger_.get(self.information_criterion)
        for i in range(self.max_depth):
            all_gains = []
            all_solvers = []
            for j, k in enumerate(possible_features):
                current_problem = problem.pivot(set(chosen_features + [k]))
                if min(current_problem.group_sizes) < 3:
                    all_gains.append(np.infty)
                    all_solvers.append(None)
                    continue
                current_solver = solver.copy()
                current_solver.fit(current_problem)
                all_gains.append(current_solver.logger_.get(self.information_criterion))
                all_solvers.append(current_solver)
            if all([gain == np.infty for gain in all_gains]):
                break
            best_gain_idx = np.argmin(all_gains)[0]
            gain_difference = all_gains[best_gain_idx] - previous_information_gain
            if gain_difference <= self.minimal_gain:
                break
            chosen_feature = possible_features[best_gain_idx]
            chosen_features.append(chosen_feature)
            possible_features = possible_features[possible_features != chosen_feature]
            solver = all_solvers[best_gain_idx]
        self.coef_ = {"chosen_categorical_features": set(chosen_features)}
        self.model = solver

    def predict(self, x):
        problem = LinearLMEProblem.from_x_y(x).pivot(self.coef_["chosen_categorical_features"])
        y = np.concatenate(self.model.predict(problem), axis=0)
        return y



