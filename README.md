# Auto-modelling
### by Cai Ytsma (cai@caiconsulting.co.uk)
Programs to prepare, train, and test datasets for multivariate regression. Designed for spectroscopy data. All algorithms from [scikit-learn](https://scikit-learn.org/stable/index.html).

## Setup
If you don't have Anaconda already, you can [download it for free](anaconda.com).

From Anaconda Prompt command line:
### Initial setup
1. Create new conda environment (you can call it something other than 'auto'): `conda create -n auto`
2. Activate the environment:	`conda activate auto`
3. Navigate to path containing files: `cd your_path_here`
4. Install required packages: `conda install --file requirements.txt`

### To run files
1. Activate the environment: `conda activate auto`
2. Navigate to path containing files: `cd your_path_here`
3. Run files: `python file_name_here.py`

## Model descriptions
### Linear regressions
- [Ordinary Least Squares (OLS)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Orthogonal Matching Pursuit (OMP)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html)
- [the Least Absolute Shrinkage and Selection Operator (LASSO)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [Ridge regression (Ridge)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Partial Least Squares (PLS)](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
- [linear Support Vector Regression (SVR-lin)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

### Non-linear regressions
- [Principal Components Regression (PCR)](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html)
- [polynomial Support Vector Regression (SVR-py)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [Random Forest regressor (RF)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Gradient Boosting regressor (GBR)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [k-Nearest Neighbors regressor (kNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
