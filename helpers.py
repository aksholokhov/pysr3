# This code implements different skmixed's subroutines.
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


def get_per_group_coefficients(beta, us, labels):
    """
    Derives per group coefficients from the vectors of fixed and per-cluster random effects.

    Parameters
    ----------
    beta: np.ndarray, shape=(n,), n is the number of fixed effects.
        Vector of fixed effects.
    us: np.ndarray, shape=(m, k), m groups, k random effects.
        Array of random effects.
    labels: np.ndarray[int], shape=(t,), t -- number of columns in the dataset INCLUDING INTERCEPT.
        Vector of labels of the column's dataset, including intercept. Labels can be the following integers:
            0 : Groups labels (ignored).
            1 : Fixed effect.
            2 : Random effect.
            3 : Both fixed and random effect.
            4 : Standard deviations for measurement errors for answers (ignored).

    Returns
    -------
    per_group_coefficients: np.ndarray, shape=(m, t)
        Array of cluster coefficients: m clusters times t coefficients.
    """

    num_studies = us.shape[0]

    per_group_coefficients = np.zeros((num_studies, len(labels)))

    for i, u in enumerate(us):
        fixed_effects_counter = 0
        random_effects_counter = 0

        for j, label in enumerate(labels):
            if label == 1:
                per_group_coefficients[i, j] = beta[fixed_effects_counter]
                fixed_effects_counter += 1
            elif label == 2:
                per_group_coefficients[i, j] = u[random_effects_counter]
                random_effects_counter += 1
            elif label == 3:
                per_group_coefficients[i, j] = beta[fixed_effects_counter] + u[random_effects_counter]
                fixed_effects_counter += 1
                random_effects_counter += 1
            else:
                continue
    return per_group_coefficients
