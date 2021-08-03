---
title: 'skmixed: A Python Package for Feature Selection in Mixed-Effect Models'
tags:
  - Python
  - feature selection
  - mixed-effect models
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
cohort studies, longitudinal data analysis, and meta-analysis.

Feature selection for LME models is harder than for linear regression models due to (a) high non-linearity, (b)
correlations between objects in the same group, and (c) presence of two related and qualitatively different types of
effects to select from: fixed and random effects. These challenges preclude straightforward application and extension of
classical feature selection methods to feature selection for LMEs.

While feature selection for LMEs is a hot topic with a lot of research [@Buscemi2019Survey], there is no universal
approach that would outperform all other strategies in every setting, leading to many suggested approaches and
corresponding implementations. Current state of the art in the field custom-tailor their numerical implementation to the
specific method that they propose. As a result, libraries for use by practitioners are either not provided, or, when
available, tightly coupled to a particular approach. This gap in open source implementation effectively blocks the
practitioner from comparing available methods, without (re)implementing complex methods in published papers.

Here, we fill this gap by (1) creating universal solvers that work with most popular regularization techniques, and on (
2) improving the selection accuracy of all of these techniques through novel relaxation reformulations that can be
applied to essentially any regularizer.

# Statement of Need

Practitioners need easy-to-use libraries that allow testing and comparisons of multiple regularization approaches on
their data, so they can choose the most effective method for their settings. These libraries should be powerful enough
to accommodate future work, for example newly proposed regularizers, without forcing the practitioner to re-implement
numerical solvers.

The skmixed package implements a variety of feature selection methods via a universal an easy-to-use interface. The
package includes widely used methods and regularizers (LASSO, Adaptive LASSO, SCAD), as well as recently developed
regularizers that have not been used before in LME feature selection(CAD, L0). The library is built to make it as easy
as possible to add new loss functions, constraints, information criteria, and regularization strategies. As part of
standard output, skmixed provides the list of features that have been selected, as well as their ranking.

The skmixed package also offers an implementation of Sparse Relaxed Regularized Regression (SR3) -- a relaxation that
boosts performance of most regularization strategies, supporting both existing and prospective ones alike. Compared to
established feature selection packages available in `R`, such as `glmmLasso` [@schelldorfer2014glmmlasso] that
implements only fitting a LASSO model for a given regularization constant, skmixed offers higher versatility of options
and a black-box interface that removes the burden of choosing hyper-parameters and information criteria from
practitioners. To the best of our knowledge, there are no standalone Python or `R` packages for mixed-effect feature
selection. For researchers, the basic implementations add value by serving as powerful baseline solutions of many
current and new regularizers, whereas their SR3-empowered versions offer state-of-the-art performance to stimulate
development of even better approaches and supporting new research.

# Core idea and structure of skmixed

At its core, skmixed uses proximal gradient descent (PGD) method as its numerical solver. This method splits the
loss-function into a smooth and a non-smooth parts, and needs only the gradient of the smooth part and the proximal
operator of the nonsmooth part to work. The smooth part typically includes the likelihood of the data, whereas the
non-smooth part captures the sparsity-promoting regularizer. For many widely-used regularizers the proximal operators
are known in a closed form ([@Zheng2019]), which is a primary source of skmixed's flexibility. PGD together with a
regularized likelihood already offer good performance for many real applications.

![Summary of skmixed method [@LINK].\label{fig:RENT}](images/summary_picture.png)

In addition to baseline implementations using PGD, skmixed also offers the SR3-relaxation to boost selection accuracy.
SR3 preconditions the likelihood by relaxing the original formulation, applying partial minimization to the smooth
piece, and then using PGD on the resulting value function. The effect is of smoothing the level-sets of the likelihood
while preserving the structure of the problem so that features can be more effectively selected. As a result, PGD on the
transformed problem converges in fewer iterations, and the variable selection process in the transformed space has
demonstrably higher selection accuracy across a wide range of regularizers.

Oracle functionality, such as likelihood, their gradients, and hessians, are implemented in `LMEOracle`. The
functionality of SR3 is implemented in a child class `LMEOracleSR3`. Regularizers are implemented in the
module `skmixed.regularizers`, and PGD is implemented as `PGDSolver` class. All 10 applications (5 regularizers and
their SR3 variantes) are implemented as a combination of these elements. More details are provided in the documentation.

For users' convenience, the black-box function `select_features` is provided. First, it initializes a model fixing most
hyper-parameters. Next, it produces a set of repeated fits with different sparsity levels. Finally, it uses Bayes
Information Criterion for choosing the ultimate model. Additionally, all fixed effects that have more than 5% posterior
probability of changing the sign are removed before the final subset of chosen fixed and random effects is provided.

In addition to the subset of optimal features, skmixed produces a diagram of inclusion-exclusion for the features for
different levels of sparsity, as well as the respective loss values, RMSEs, and explained variances. For linear algebra
and visualization functionality, skmixed uses numpy and matplotlib respectively.

# Ongoing Research and Dissemination

The manuscript "Feature Selection Methods for Linear Mixed Effects Models" is undergoing a peer-review process.
Currently, skmixed is used for improving reliability of meta-research studies in Institute for Health Metrics and
Evaluations in the University of Washington.

# References
