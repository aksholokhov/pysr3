#     skmixed: Library for Feature Selection in Linear Mixed-Effect Models
#     Copyright (C) 2020  Aleksei Sholokhov
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "__title__", "__summary__", "__uri__", "__version__", "__author__",
    "__email__", "__license__", "__copyright__"
]

__title__ = "PySR3"
__summary__ = "Python Library for Sparse Relaxed Regularized Regression."
__long_description__ = ("This package implements classic and novel feature selection algorithms " +
                        " for linear and mixed-effect models." +
                        " It supports many widely used regularization techniques, like LASSO, A-LASSO, CAD and SCAD." +
                        " See README.md for details and examples.")
__uri__ = "https://github.com/aksholokhov/pysr3"
__classifiers__ = [
                      "Programming Language :: Python :: 3",
                      'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                      "Operating System :: OS Independent",
                  ],

__version__ = "0.3.0"

__author__ = "Aleksei Sholokhov"
__email__ = "aksh@uw.edu"

__license__ = "GNU GPLv3"
__copyright__ = f"Copyright 2020-2021 {__author__}"
