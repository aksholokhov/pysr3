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
datasets include more information that may be related to key research questions. However, in predictive modeling,
the presence of irrelevant features can hamper stability of estimation, validity of inference, prediction accuracy, and qualitative conclusions of the research. Reliable feature selection is therefore key in a wide range of regression settings.

While feature selection has been a hot topic with a lot of research [@Buscemi2019Survey, @miao2016survey, @li2020survey],
there is no known universal approach that would outperform all other strategies in every setting, leading to many suggested approaches and
corresponding implementations. Current state of the art in the field custom-tailor their numerical implementation to the
specific loss and the method that they propose. As a result, libraries for use by practitioners are either not provided, or, when
available, tightly coupled to a particular approach. This gap in open source implementation effectively blocks the
practitioner from comparing available methods, without (re)implementing complex methods in published papers.

In addition, very few methods are equipped to handle ill-conditioned problems. Ill-conditioning naturally arises
in real-world datasets due to the presence of correlated and collinear predictors, outliers, and missing values. Poor conditioning 
adversely affects stability of numerical solvers and limits the maximum size of the datasets that the model can learn from. Therefore, 
robust solvers that work for a wide spectrum of models are desired.

Here, we fill this gap by creating universal solvers that (1) work with most popular regularized regression techniques, 
and (2) improve the selection accuracy of all of these techniques through novel relaxation reformulations that can be
applied to essentially any loss and regularizer.

# Statement of Need
   
Practitioners need easy-to-use libraries that allow testing and comparisons of multiple regularization approaches on
their data, so they can choose the most effective feature selection method for their settings. These libraries should be powerful enough
to accommodate future work, for example newly proposed regularizers, without forcing the practitioner to re-implement
numerical solvers. At the same time, these universal solvers must effectively handle ill-conditioned problems,
that are very common in raw real-world datasets.

 
The library is built to make it as easy as possible to add new loss functions, constraints, 
information criteria, and regularization strategies. Moreover, all PySR3 models fully adhere the standards and 
interface requirements of `sklearn` ([@sklearn_api]), providing a familiar interface to users. 

Linear Mixed-Effects (LME) models can be viewed as extensions of standard linear regression to clustered data settings.
They are a fundamental tool for modeling between-cluster variations of the quantity of interest, and commonly arise in
cohort studies, longitudinal data analysis, and meta-analysis. Feature selection for linear mixed-effects models
is harder than for linear regression models due to (a) high non-linearity, (b) correlations between objects 
in the same group, and (c) presence of two related and qualitatively different types of
effects to select from: fixed and random effects. 
To the best of our knowledge, there are no standalone Python packages for mixed-effect feature selection.

# Core idea and structure of pysr3

At its core, PySR3 uses proximal gradient descent (PGD) method as its numerical solver. This method splits the
loss-function into a smooth and a non-smooth parts, and needs only the gradient of the smooth part and the proximal
operator of the non-smooth part to work. The smooth part typically includes the likelihood of the data, whereas the
non-smooth part captures the sparsity-promoting regularizer. For many widely-used regularizers the proximal operators
are known in a closed form ([@Zheng2019]). PGD together with a regularized likelihood already offer 
good performance for many real applications.

In addition to baseline PGD models, PySR3 also offers their SR3-relaxations.
SR3 preconditions the likelihood by relaxing the original formulation, applying partial minimization to the smooth
piece, and then using PGD on the resulting value function. The effect is of smoothing the level-sets of the likelihood
while preserving the structure of the problem so that features can be more effectively selected. As a result, PGD on the
transformed problem converges in fewer iterations, and the variable selection process in the transformed space has
demonstrably higher selection accuracy and lower rate of false positives across a wide range of regularizers for both linear ([@Zheng2019]) 
and LME (?) models.

![Summary of PySR3 method.\label{fig:lme_summary}](images/summary_picture.png)

Because of full compatibility with `sklearn`, all `pysr3` models can be used in pipeline with classic modelling 
blocks such as data pre-processors, randomized grid search, cross-validation, and quality metrics. 

More information about the structure of the library can be found in [documentation](), whereas the mathematical contributions 
are extensively discussed in [@Zheng2019] for linear and in [?] for linear mixed-effects models respectively.


# Ongoing Research and Dissemination

The manuscript "Feature Selection Methods for Linear Mixed Effects Models" is undergoing a peer-review process.
Since its introduction in [@Zheng2019], SR3 has been used in multiple applications,
such as model discovery ([@Mendible2020]), optimal dose management ([@levin2019proof]) and others(?).
Currently, PySR3 is used for improving reliability of meta-research studies at Institute for Health Metrics and
Evaluations, University of Washington.

# References
