# This code implements solvers for linear mixed-effects models.
# Copyright (C) 2020 Aleksei Sholokhov, aksh@uw.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np

from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score

from skmixed.lme.models import LinearLMESparseModel
from skmixed.lme.problems import LinearLMEProblem


class Tree(BaseEstimator, RegressorMixin):
    def __init__(self, model: LinearLMESparseModel, max_depth=3, information_criterion="IC_vaida2005aic",
                 minimal_gain=1e1):
        self.model = model
        self.max_depth = max_depth
        self.information_criterion = information_criterion
        self.minimal_gain = minimal_gain

    def fit_problem(self, problem):
        chosen_features = [0]
        possible_features = np.arange(1, problem.num_categorical_features)
        problem = problem.pivot((0,))
        model = clone(self.model)
        model.logger_keys = model.logger_keys + (self.information_criterion,)
        model.fit_problem(problem)

        previous_information_gain = model.logger_.get(self.information_criterion)
        for i in range(self.max_depth):
            all_gains = []
            all_solvers = []
            for j, k in enumerate(possible_features):           # Tree complexity stop
                current_problem = problem.pivot(set(chosen_features + [k]))
                if min(current_problem.groups_sizes) < 3:
                    all_gains.append(np.infty)
                    all_solvers.append(None)
                    continue
                current_solver = clone(model)
                current_solver.coef_ = model.coef_
                current_solver.fit_problem(current_problem, warm_start=True)
                all_gains.append(current_solver.logger_.get(self.information_criterion))
                all_solvers.append(current_solver)
            if all([gain == np.infty for gain in all_gains]):    # Granularity stop
                break
            best_gain_idx = np.argmin(all_gains)
            gain_difference = previous_information_gain - all_gains[best_gain_idx]
            if gain_difference <= self.minimal_gain:            # Information gain stop
                break
            chosen_feature = possible_features[best_gain_idx]
            chosen_features.append(chosen_feature)
            possible_features = possible_features[possible_features != chosen_feature]
            model = all_solvers[best_gain_idx]
            previous_information_gain = all_gains[best_gain_idx]

        self.coef_ = {"chosen_categorical_features": set(chosen_features)}
        self.fitted_model_ = model


    def predict(self, x):
        # check_is_fitted(self, 'coef_')    # TODO: figure out why this check does not work
        problem = LinearLMEProblem.from_x_y(x)
        return self.predict_problem(problem)

    def predict_problem(self, problem):
        # check_is_fitted(self, 'coef_')    # same
        pivoted_problem = problem.pivot(self.coef_["chosen_categorical_features"])
        return  self.fitted_model_.predict_problem(pivoted_problem)


if __name__ == "__main__":
    problem, true_parameters = LinearLMEProblem.generate(groups_sizes=[40, 30, 50],
                                                        features_labels=[3, 3, 6, 5, 5, 6],
                                                        random_intercept=True,
                                                        obs_std=0.1,
                                                        seed=0)
    continuous_model = LinearLMESparseModel(lb=0, lg=0)
    tree_model = Tree(model=continuous_model)
    tree_model.fit_problem(problem)
    y_pred = tree_model.predict_problem(problem)
    x, y_true = problem.to_x_y()
    explained_variance = explained_variance_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    a = 3
    pass