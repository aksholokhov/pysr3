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
Feature selection is a core step in regression studies. With new data acquisition tools available, modern datasets are including more information about the observed phenomena. Not all this information is relevant or allows reliable extrapolation. In fact, the presence of irrelevant features may hamper both  stability of the numerical solvers, prediction accuracy, and qualitative conclusions of research. Hence, reliable tools for feature selection in regression settings are necessary.

Linear Mixed-Effects (LME) models are an extension of a standard linear regression to clustered data settings. They are a fundamental tool for modelling per-cluster variations of the quantity of interest in cohort studies, longitudinal data analysis, and meta-analysis.

Feature selection for linear mixed-effect models is harder than for linear regression models due to (a) high non-linearity, (b) correlations between objects of the same group, and (c) presence of two qualitatively different but intertwined groups of effects to select from, known as fixed and random effects. These reasons prevent classical feature selection methods from being applied to  mixed-effect problems in a straightforward fashion. 

Although many adaptations were suggested in literature [@Buscemi2019Survey], there is still no universal approach that would outperform all other strategies in every setting. At the same time, most of the proposed methods custom-tailor their numerical implementation to the sparsity-promoting regularization method that they use. Hence, the provided libraries are often locked to a particular statistical approach, when at all available. As a result, it effectively prohibits the practitioner from comparing the methods due to time and work burdens of implementing them as separate coding projects.

We focus our efforts on (1) creating universal solvers that work with most popular regularization techniques, and on (2) improving the selection accuracy of these techniques that work regardless of which regularizer is used.  

# Statement of Need
Practitioners need easy-to-use libraries that would allow testing and comparing various regularization approaches on their data. Moreover, these libraries should be powerful enough to accommodate yet-to-be-created regularizers without forcing the practitioner to re-implement numerical solvers.

The skmixed package implements a variety of feature selection methods via a universal an easy-to-use interface. The package includes widely-used (LASSO, Adaptive LASSO, SCAD), as well as niche (CAD), and novel ($\ell_0$) regularizers to choose from. The library is built to make it as easy as possible to add new loss functions, constraints, information criteria, and regularization strategies. As an output, skmixed provides the list of features that have been selected, as well as their ranking.

 The library also offers an implementation of Sparse Relaxed Regularized Regression (SR3) -- a relaxation that boosts performance of most regularization strategies, supporting both existing and prospective ones alike. Compared to established feature selection packages available in `R`, such as `glmmLasso` [@schelldorfer2014glmmlasso] that implements only fitting a LASSO model for a given regularization constant, skmixed offers higher versatility of options and a black-box interface that removes the burden of choosing hyper-parameters and information criteria from practitioners. To the best of our knowledge, there are no standalone Python packages for mixed-effect feature selection. Besides, for researchers, the basic implementations possess value by being great baseline solutions, whereas their SR3-empowered versions offer state-of-the-art performance to stimulate development of even better algorithms.

# Core idea and structure of skmixed

At its core, skmixed uses proximal gradient descent (PGD) method as its numerical solver. This method splits the loss-function into a smooth and a non-smooth parts, and requires the gradient of the former and the proximal operator of the latter, to work. The smooth part typically includes the likelihood of the data, whereas the non-smooth part normally represents sparsity-promoting regularizer. For many widely-used regularizers the proximal operators are known in a closed form ([@Zheng2019]), which is a primary source of skmixed's flexibility. 

![Summary of skmixed method [@LINK].\label{fig:RENT}](images/summary_picture.png)

Whereas PGD together with a regularized likelihood offer decent performance for many real applications, skmixed also offers SR3-relaxation to boost the selection accuracy. SR3 transforms the likelihood smoothing the level-sets while keeping nearly the same minima. As a result, PGD convergence in fewer iterations and offers higher selection accuracy across a wide board of regularizers.

The oracles functionality, such as likelihood, their gradients, and hessians, are implemented in `LMEOracle`. The functionality of SR3 is implemented in a child class `LMEOracleSR3`. Regularizers are implemented in the module `skmixed.regularizers`, and PGD is implemented as `PGDSolver` class. All 10 are implemented as a combination of these elements. More details are provided in the documentation.
 
For users' convenience, the black-box function `select_features` is provided. First, it initializes a model fixing most hyper-parameters. Next, it produces a set of repeated fits with different sparsity levels. Finally, it uses Bayes Information Criterion for choosing the ultimate model. Additionally, all fixed effects that have more than 5% posterior probability of changing the sign are removed before the final subset of chosen fixed and random effects is provided.

In addition to the subset of optimal features, skmixed produces a diagram of inclusion-exclusion for the features for different levels of sparsity, as well as the respective loss values, RMSEs, and explained variances. For linear algebra and visualization functionality, skmixed uses numpy and matplotlib respectively.

# Ongoing Research and Dissemination
The manuscript "Feature Selection Methods for Linear Mixed Effects Models" is undergoing a peer-review process. Currently, skmixed is used for improving reliability of meta-research studies in Institute for Health Metrics and Evaluations in the University of Washington.

# References
