---
title: 'pysr3: A Python Package for Sparse Relaxed Regularized Regression'
tags:
  - Python
  - feature selection
  - linear models
  - mixed-effect models
  - regularization
authors:
  - name: Aleksei Sholokhov
    orcid: 0000-0001-8173-6236
    affiliation: 1
  - name: Peng Zheng
    orcid: 0000-0003-1327-4855
    affiliation: 2
  - name: Aleksander Aravkin
    orcid: 0000-0001-5210-132X
    affiliation: "1, 2"
affiliations:
 - name: Department of Applied Mathematics, University of Washington
   index: 1
 - name: Department of Health Metrics Sciences, University of Washington
   index: 2
date: 30.7.2021
bibliography: paper.bib

---

# Summary

Feature selection is a core step in regression modeling. As new types of data becomes readily available, current
datasets include more information that may be related to key research questions. In most contexts, only some of
information is relevant or allows reliable extrapolation. For predictive modeling, the presence of irrelevant features
can hamper stability of estimation, validity of inference, prediction accuracy, and qualitative conclusions of the
research. Reliable feature selection is therefore key in a wide range of regression settings.

Linear Mixed-Effects (LME) models can be viewed as extensions of standard linear regression to clustered data settings.
They are a fundamental tool for modeling between-cluster variations of the quantity of interest, and commonly arise in
cohort studies, longitudinal data analysis, and meta-analysis. Feature selection for linear mixed-effects models is harder than for linear regression models due to (a) high non-linearity, (b) correlations between objects in the same group, and (c) presence of two related and qualitatively different types of
effects to select from: fixed and random effects. These challenges preclude straightforward application and extension of
classical feature selection methods to feature selection for LMEs.

While feature selection for linear and mixed-effects models is a hot topic with a lot of research [@Buscemi2019Survey, @miao2016survey, @li2020survey],
there is no known universal approach that would outperform all other strategies in every setting, leading to many suggested approaches and
corresponding implementations. Current state of the art in the field custom-tailor their numerical implementation to the
specific loss and the method that they propose. As a result, libraries for use by practitioners are either not provided, or, when
available, tightly coupled to a particular approach. This gap in open source implementation effectively blocks the
practitioner from comparing available methods, without (re)implementing complex methods in published papers.

At the same time, feature selection methods are routinely applied to ill-conditioned problems. Ill-conditioning naturally arises
in real-world datasets due to the presence of correlated and collinear predictors, outliers, and missing values. Poor conditioning 
adversely affects stability of numerical solvers, forecasting and prediction capabilities, and, consequently, qualitative results of analysis.
Despite tremendous advancements in methodology, there is still a lack of fast universal solvers for large-scale ill-conditioned regression problems 
that work with arbitrary non-convex loss functions and non-smooth non-convex regularizers.

Here, we fill this gap by (1) creating universal solvers that work with most popular regularized regression techniques, 
and on (2) improving the selection accuracy of all of these techniques through novel relaxation reformulations that can be
applied to essentially any loss and regularizer.

# Statement of Need
   
Practitioners need easy-to-use libraries that allow testing and comparisons of multiple regularization approaches on
their data, so they can choose the most effective feature selection method for their settings. These libraries should be powerful enough
to accommodate future work, for example newly proposed losses and regularizers, without forcing the practitioner to re-implement
numerical solvers. The solvers must effectively handle ill-conditioned problems, that are very common in raw real-world datasets.

PySR3 package implements a variety of feature selection methods for linear and linear mixed-effects regression via a universal an easy-to-use interface. These methods use Sparse Relaxed Regularized Regression (SR3) -- a relaxation technique that boosts performance of most regularization strategies in presence of ill-conditioned data. This technique is regularizer- and loss-agnostic, and can be applied to a wide class of models ([@Zheng2020]).
The package includes widely used regularizers (LASSO, Adaptive LASSO, SCAD), as well as recently developed
ones (CAD, L0). The library is built to make it as easy as possible to add new loss functions, constraints, 
information criteria, and regularization strategies. Moreover, all PySR3 models fully adhere the standards and 
interface requirements of `sklearn` ([@sklearn_api]), providing an interface that is familiar to the modelling community.

To the best of our knowledge, there are no standalone Python packages for mixed-effect feature selection. Compared to established feature selection packages available in `R`, such as `glmmLasso` [@schelldorfer2014glmmlasso] that implements only fitting a LASSO model for a given regularization constant, PySR3 offers higher versatility of options and black-box toolkit that removes the burden of choosing hyper-parameters and information criteria from practitioners. For researchers, the basic implementations add value by serving as powerful baseline solutions of many
current and new regularizers, whereas their SR3-empowered versions offer state-of-the-art performance to stimulate
development of even better approaches and supporting new research.


# Core idea and structure of pysr3

At its core, PySR3 uses proximal gradient descent (PGD) method as its numerical solver. This method splits the
loss-function into a smooth and a non-smooth parts, and needs only the gradient of the smooth part and the proximal
operator of the nonsmooth part to work. The smooth part typically includes the likelihood of the data, whereas the
non-smooth part captures the sparsity-promoting regularizer. For many widely-used regularizers the proximal operators
are known in a closed form ([@Zheng2019]), which is a primary source of the package's flexibility. PGD together with a
regularized likelihood already offer good performance for many real applications.

In addition to baseline implementations using PGD, PySR3 also offers their SR3-relaxations.
SR3 preconditions the likelihood by relaxing the original formulation, applying partial minimization to the smooth
piece, and then using PGD on the resulting value function. The effect is of smoothing the level-sets of the likelihood
while preserving the structure of the problem so that features can be more effectively selected. As a result, PGD on the
transformed problem converges in fewer iterations, and the variable selection process in the transformed space has
demonstrably higher selection accuracy and lower rate of false positives across a wide range of regularizers for both linear ([@Zheng2019]) 
and LME (?) models.

![Summary of PySR3 method.\label{fig:lme_summary}](images/summary_picture.png)

Oracle functionality, such as likelihood, their gradients, and hessians, are implemented in `LMEOracle`. The
functionality of SR3 is implemented in a child class `LMEOracleSR3`. Regularizers are implemented in the
module `PySR3.regularizers`, and PGD is implemented as `PGDSolver` class. All 10 applications (5 regularizers and
their SR3 variantes) are implemented as a combination of these elements. More details are provided in the documentation.

For users' convenience, the black-box function `select_features` is provided. First, it initializes a model fixing most
hyper-parameters. Next, it produces a set of repeated fits with different sparsity levels. Finally, it uses Bayes
Information Criterion (BIC) for choosing the ultimate model. Additionally, all fixed effects that have more than 5% posterior
probability of changing the sign are removed before the final subset of chosen fixed and random effects is provided.

In addition to the subset of optimal features, PySR3 produces a diagram of inclusion-exclusion for the features for
different levels of sparsity, as well as the respective loss values, RMSEs, and explained variances. For linear algebra
and visualization functionality, PySR3 uses `numpy` and `matplotlib` respectively.

# Ongoing Research and Dissemination

The manuscript "Feature Selection Methods for Linear Mixed Effects Models" is undergoing a peer-review process.
SR3 has been used in multiple applications, such as model discovery ([@Mendible2020]), optimal dose management ([@levin2019proof]) and others(?).
Currently, PySR3 is used for improving reliability of meta-research studies at Institute for Health Metrics and
Evaluations, University of Washington.

# References
