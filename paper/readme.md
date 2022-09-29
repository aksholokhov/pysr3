# Datasets for [Sholokhov et. al. 2022 "A Relaxation Approach to Feature Selection for Linear Mixed Effects Models"](https://arxiv.org/abs/2205.06925?context=stat)

This folder contains datasets that we used in Chapter 4 of our paper. 

## Chapter 4.1: Experiments on synthetic data 
The folder `synthetic_data` contains 100 CSV tables. The rows are observations (objects). Each table has the following columns:

* `group` -- which group does this object belong to 
* `target` -- target varable a.k.a. observations a.k.a. `y`
* `variance` -- variance of observation noise 
* `fixed`, `random`, or `fixed+random` -- features a.k.a. covariates. 

The names of feature columns indicate whether they factor into the model as `fixed` effects, `random` effects, or both (`fixed+random`).

## Chapter 4.2 Bullying Data
The data from Chapter 4.2 is in `bullying_data.csv` file. See Appendix B.2 for the description and the meaning of features, as well as the [original GBD study](https://www.healthdata.org/results/gbd_summaries/2019/bullying-victimization-level-3-risk).