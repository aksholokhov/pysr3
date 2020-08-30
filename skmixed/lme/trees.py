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
        x, y = problem.to_x_y()
        model.fit(x, y)  # TODO: fix it back to fit_problem

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
            # assert len(best_gain_idx) == 1, "More than one best feature option"
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


class Forest(BaseEstimator, RegressorMixin):

    def __init__(self, continuous_model, num_trees=10, max_depth=3, information_criterion="IC_vaida2005aic",
                 minimal_gain=1e1):
        self.model = continuous_model
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.information_criterion=information_criterion
        self.minimal_gain=minimal_gain

    def fit_problem(self, problem: LinearLMEProblem, seed=42):
        self.trees_ = []
        self.bootstrap_idxs_ = []
        assert problem.answers is not None, "Problem does not include the target variable"
        for i in range(self.num_trees):
            bootstrap_problem = problem.bootstrap(seed=i+seed)
            tree = Tree(model=clone(self.model),
                        max_depth=self.max_depth,
                        information_criterion=self.information_criterion,
                        minimal_gain=self.minimal_gain)
            tree.fit_problem(bootstrap_problem)
            self.trees_.append(tree)
            self.bootstrap_idxs_.append(bootstrap_problem.categorical_features_bootstrap_idx)

    def get_ensemble_predictions(self, problem: LinearLMEProblem):
        # check_is_fitted(self, "trees_", "bootstrap_idxs_")
        predictions = []
        for tree, bootstrap_idxs in zip(self.trees_, self.bootstrap_idxs_):
            bootstrap_problem = problem.bootstrap(categorical_features_idx=bootstrap_idxs, do_bootstrap_objects=False)
            answers = tree.predict_problem(bootstrap_problem)
            predictions.append(answers)
        return predictions

    def predict_problem(self, problem: LinearLMEProblem):
        predictions = self.get_ensemble_predictions(problem)
        return np.mean(predictions, axis=0)

    def get_prediction_uncertainty(self, problem, percentile=5):
        assert 0 < percentile < 50, "Percentile should be between 0 and 50"
        predictions = np.array(self.get_ensemble_predictions(problem))
        lower_percentile = np.percentile(predictions, axis=0, q=percentile)
        higher_percentile = np.percentile(predictions, axis=0, q=100-percentile)
        return lower_percentile, higher_percentile
