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

## Program descriptions
#### stratify_samples.py
Assigns *k* folds to input data by sorting samples by abundance per variable, grouping by `Sample Name` column (if applicable), and sequentially assigning folds to maintain compositional diversity per subset. *k* is user-input or defaults to 5, as recommended by [Dyar and Ytsma (2021)](https://doi.org/10.1016/j.sab.2021.106073).
##### Input prompt
- metadata .csv file path
##### Output
- metadata .csv file with new `{var}_Folds` column per variable and `_stratified` appended to original filename

#### spectral_regression_modelling.py
The main program, this prompts the user for the input spectra and metadata (example datasets in `\example data`), automatically identifies relevant variables to calibrate, and prompts user to specify which regression models to train with. Users can optionally assign one of the folds as a test dataset. The model and test set choices can be standardized for all variables or chosen per variable. The program automatically optimizes models, where possible, using a custom grid search and trains on the optimum model per method.  Once trained, the program also outputs predicted versus values as a .csv and scatter plot. For linear models, coefficients are exported as a .csv and a plot of their weights over an example spectrum.
##### Input prompts
- Data folder path
- Spectra filename
- Metadata filename
- Whether to standardize procedure for all variables (where applicable)
- Whether to use a fold as a test dataset; if so, which fold
- Which modelling method(s) to train with (all, one, or combination)
##### Output
- Model file
    - `{variable}_{model}_model.asc`
- Model-predicted versus true values and XY plots 
    - `{variable}_{model}_{train/test}_pred_true.csv`
    - `{variable}_{model}_{train/test}_pred_true_plot.jpg`
    - `{variable}_{model}_{train/test}_pred_true_plot.eps`
- Model coefficient values and plot over example spectrum
     - `{variable}_{model}_coefs.csv`
     - `{variable}_{model}_coefs_plot.jpg`
     - `{variable}_{model}_coefs_plot.eps`
- Overall results (# train/test, model parameters, RMSEs, R2s)
     - `modelling_train_results.csv` if no folds were used for testing
     - `modelling_train_test_results.csv` if a fold was used for testing

#### test_model.py
Tests models on input data. Optionally include metadata for test samples to generate predicted versus true scatter plots and accuracy metrics. Model files must be in correct format:
  - .asc from scikitlearn algorithms (e.g. output from `spectral_regression_modelling.py`)

##### Input prompts
  - Model file path
  - Test spectra file path
  - If there is test metadata; if so, the metadata file path

##### Outputs
  - Predicted values
    - `{variable}_{method}_test_pred.csv` or `{variable}_{method}_test_pred_true.csv` if have metadata
  - Predicted vs. true scatter plot (if have metadata)

#### merge_model_results.py
Merges result files `from spectral_regression_modelling.py` contained within subfolders into one .csv. Has only been tested for files residing one level down from the input folder, where each result file has the same format (i.e all `modelling_train_results.csv`, or all `modelling_train_test_results.csv`). Adds column `folder` with the name of each subfolder to distinguish entries.
##### Input prompt
- Folder with subfolders containing model result files
##### Output
- Compiled .csv with `compiled_` appended to beginning of filename

#### model_tools.py
Helper file that contains many of the functions used by the above programs.

## Calibration models
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
