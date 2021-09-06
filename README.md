![](https://img.shields.io/pypi/l/pysr3)
![](https://img.shields.io/pypi/v/pysr3)
![](https://img.shields.io/github/workflow/status/aksholokhov/pysr3/Testing%20and%20Coverage/sr3)
[![](https://img.shields.io/badge/docs-up-green)](https://aksholokhov.github.io/pysr3/)
![](https://img.shields.io/codecov/c/github/aksholokhov/pysr3/sr3?flag=unittests)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/749695b3c6fd43bb9fdb499ec0ace67b)](https://www.codacy.com/gh/aksholokhov/pysr3/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=aksholokhov/pysr3&amp;utm_campaign=Badge_Grade)

# Quickstart with `pysr3`

SR3 is a relaxation method designed for accurate feature selection.
It currently supports:

* Linear Models (LASSO, A-LASSO, CAD, SCAD)
* Linear Mixed-Effect Models (L0, LASSO, A-LASSO, CAD, SCAD)

## Installation

pysr3 can be installed via
```bash
 pip install pysr3
```

## Requirements
Make sure that Python 3.6 or higher is installed. The package has the following
dependencies, as listed in requirements.txt:

* numpy>=1.21.1
* pandas>=1.3.1
* scipy>=1.7.1
* PyYAML>=5.4.1
* scikit_learn>=0.24.2

## Usage
pysr3 models are fully compatible to [sklearn standards](https://scikit-learn.org/stable/developers/develop.html),
so you can use them as you normally would use a sklearn model.

### Linear Models
A simple example of using SR3-empowered LASSO for feature selection is shown below.


```python
import numpy as np

from pysr3.linear.problems import LinearProblem

# Create a sample dataset
seed = 42
num_objects = 300
num_features = 500
np.random.seed(seed)
# create a vector of true model's coefficients
true_x = np.random.choice(2, size=num_features, p=np.array([0.9, 0.1]))
# create sample data
a = 10 * np.random.randn(num_objects, num_features)
b = a.dot(true_x) + np.random.randn(num_objects)

print(f"The dataset has {a.shape[0]} objects and {a.shape[1]} features; \n"
      f"The vector of true parameters contains {sum(true_x != 0)} non-zero elements out of {num_features}.")
```

    The dataset has 300 objects and 500 features; 
    The vector of true parameters contains 55 non-zero elements out of 500.



```python
# Automatic features selection using information criterion
from pysr3.linear.models import LinearL1ModelSR3
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

# Here we use SR3-empowered LASSO, but many other popular regularizers are also available
# See the glossary of models for more details.
model = LinearL1ModelSR3()
# We will search for the best model over the range of strengths for the regularizer
params = {
    "lam": loguniform(1e-1, 1e2)
}
selector = RandomizedSearchCV(estimator=model,
                              param_distributions=params,
                              n_iter=50,
                              # The function below evaluates an information criterion
                              # on the test portion of CV-splits.
                              scoring=lambda clf, x, y: -clf.get_information_criterion(x, y, ic='bic'))

selector.fit(a, b)
maybe_x = selector.best_estimator_.coef_['x']
tn, fp, fn, tp = confusion_matrix(true_x, maybe_x != 0).ravel()

print(f"The model found {tp} out of {tp + fn} features correctly, but also chose {fp} extra irrelevant features. \n"
      f"The best parameter is {selector.best_params_}")
```

    The model found 55 out of 55 features correctly, but also chose 2 extra irrelevant features. 
    The best parameter is {'lam': 0.15055187290939537}


### Linear Mixed-Effects Models

Below we show how to use Linear Mixed-Effects (LME) models for simultaneous selection
of fixed and random effects.


```python
from pysr3.lme.models import L1LmeModelSR3
from pysr3.lme.problems import LMEProblem, LMEStratifiedShuffleSplit

problem, true_parameters = LMEProblem.generate(
    groups_sizes=[10] * 6,  # 6 groups, 10 objects each
    features_labels=["fixed+random"] * 20,  # 20 features, each one having both fixed and random components
    beta=np.array([0, 1] * 10),  # True beta (fixed effects) has every other coefficient active
    gamma=np.array([0, 0, 0, 1] * 5),  # True gamma (variances of random effects) has every fourth coefficient active
    obs_var=0.1  # The errors have standard errors of sqrt(0.1) ~= 0.33

)

# LMEProblem provides a very convenient representation
# of the problem. See the documentation for more details.

# It also can be converted to a more familiar representation
x, y, columns_labels = problem.to_x_y()
# columns_labels describe the roles of the columns in x:
# fixed effect, random effect, or both of those, as well as
```


```python
# We use SR3-empowered LASSO model, but many other popular models are also available.
# See the glossary of models for more details.
model = L1LmeModelSR3()

# We're going to select features by varying the strength of the prior
# and choosing the model that yields the best information criterion
# on the validation set.
params = {
    "lam": loguniform(1e-3, 1e3)
}
# We use standard functionality of sklearn to perform grid-search.
selector = RandomizedSearchCV(estimator=model,
                              param_distributions=params,
                              n_iter=10,  # number of points from parameters space to sample
                              # the class below implements CV-splits for LME models
                              cv=LMEStratifiedShuffleSplit(n_splits=2, test_size=0.5,
                                                           random_state=seed,
                                                           columns_labels=columns_labels),
                              # The function below will evaluate the information criterion
                              # on the test-sets during cross-validation.
                              # We use IC from Muller2018, but other options (AIC, BIC) are also available
                              scoring=lambda clf, x, y: -clf.get_information_criterion(x,
                                                                                       y,
                                                                                       columns_labels=columns_labels,
                                                                                       ic="muller_ic"),
                              random_state=seed,
                              n_jobs=20
                              )
selector.fit(x, y, columns_labels=columns_labels)
best_model = selector.best_estimator_

maybe_beta = best_model.coef_["beta"]
maybe_gamma = best_model.coef_["gamma"]
ftn, ffp, ffn, ftp = confusion_matrix(true_parameters["beta"], abs(maybe_beta) > 1e-2).ravel()
rtn, rfp, rfn, rtp = confusion_matrix(true_parameters["gamma"], abs(maybe_gamma) > 1e-2).ravel()

print(
    f"The model found {ftp} out of {ftp + ffn} correct fixed features, and also chose {ffp} out of {ftn + ffn} extra irrelevant fixed features. \n"
    f"It also identified {rtp} out of {rtp + rfn} random effects correctly, and got {rfp} out of {rtn + rfn} non-present random effects. \n"
    f"The best sparsity parameter is {selector.best_params_}")
```

    The model found 9 out of 10 correct fixed features, and also chose 2 out of 9 extra irrelevant fixed features. 
    It also identified 5 out of 5 random effects correctly, and got 0 out of 15 non-present random effects. 
    The best sparsity parameter is {'lam': 4.0428727350273315}

