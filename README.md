# Auto-modelling
### by Cai Ytsma (cai@caiconsulting.co.uk)
Programs to format, train, and test datasets for multivariate regression. Designed for spectroscopy data. All machine learning algorithms from [scikit-learn](https://scikit-learn.org/stable/index.html).

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

## Formatting programs
### stratify_samples.py
Assigns *k* folds to input data by sorting samples by abundance per variable, grouping by `Sample_Name` column (if present), and sequentially assigning folds to maintain compositional diversity per subset. *k* is user-input or defaults to 5, as recommended by [Dyar and Ytsma (2021)](https://doi.org/10.1016/j.sab.2021.106073). The program adds variable-specific fold columns (`{var}_Folds`) as well as a universal `Folds` column for multi-variable metadata.
##### Input prompt
- Metadata filepath
##### Output
- Transformed metadata file with `_stratified` stuffix

### spectra_first_derivative.py
Converts input spectra to their first derivative and exports the transformed data. The first column should be the axis, with a 'wave' header.
##### Input prompt
- Spectra filepath
##### Output
- Transformed spectra file with `_first_derivative` suffix

### merge_model_results.py
Merges result files `from spectral_regression_modelling.py` contained within subfolders into one .csv. Has only been tested for files residing one level down from the input folder, where each result file has the same format (i.e all `modelling_train_results.csv`, or all `modelling_train_test_results.csv`). Adds column `folder` with the name of each subfolder to distinguish entries.
##### Input prompt
- Folder with subfolders containing model result files
##### Output
- Compiled .csv with `compiled_` appended to beginning of filename

## Calibration programs
### spectral_regression_modelling.py
The main program, this prompts the user for the input spectra and metadata (example datasets in `\example data`), automatically identifies relevant variables to calibrate, and prompts user to specify which regression models to train with. Users can optionally assign one of the folds as a test dataset. The model and test set choices can be standardized for all variables or chosen per variable. The program automatically optimizes models, where possible, using a custom grid search and trains on the optimum model per method.  Once trained, the program also outputs predicted versus values as a .csv and scatter plot. For linear models, coefficients are exported as a .csv and a plot of their weights over an example spectrum.
##### Input prompts
- Data folder path
- Spectra filename
- Metadata filename
- Whether to standardize procedure for all variables (where applicable)
- Whether to use a fold as a test dataset; if so, which fold
- Which modelling method(s) to train with (all, one, or combination)
##### Output
- Model file (`{var}_{method}_model.asc`)
- Predicted vs. true values (`{variable}_{method}_{train/test}_pred_true.csv`) and scatter plots (`{var}_{method}_{train/test}_pred_true_plot.jpg` and `.eps`)
- Model coefficient values (`{variable}_{method}_coefs.csv`) and scatter plot on top of example spectrum (`{variable}_{model}_coefs_plot.jpg` and `.eps`)
- Overall model results (# train/test, model parameters, RMSEs, R2s)
     - `modelling_train_results.csv` if no folds were used for testing
     - `modelling_train_test_results.csv` if a fold was used for testing

### test_model.py
Tests models on input data. Optionally include metadata for test samples to generate predicted versus true scatter plots and accuracy metrics. Model files are assumed to be `.asc` files from Scikit-learn algorithms (e.g., output from `spectral_regression_modelling.py`).

##### Input prompts
  - Model filepath
  - Test spectra filepath
  - Test metadata filepath (optional)

##### Outputs
  - Predicted values (`{var}_{method}_test_pred.csv`)
  - If have metadata, predicted vs. true values (`{var}_{method}_test_pred_true.csv`) and plots (`{var}_{method}_test_pred_true_plot.jpg` and `.eps`)

### PLS2_modelling.py
A program that runs only PLS2 regression, which predict multiple variables in one model. File formats match `spectral_regression_modelling`, except that all variable columns in the metadata file will be predicted by the model. The user can see the results of cross-validation per variable as well as the averaged result to decide which PLS2 component to choose for the overall model.
##### Input prompts
- Folder path
- Spectra file name
- Metadata file name
- Maximum number of PLS components (default=30)
##### Outputs 
- Comparison plots of cross-validation results per variable and for the average (`PLS2_RMSECV_plots_{variables}.jpg`)
- Single PLS2 model (`PLS2_model_{variables}.asc`)
- Training results for overall model and for each variable (`PLS2_train_results_{variables}.csv`)
- Predicted vs. true values (`PLS2_train_pred_true_{variables}.csv`) and scatter plots (`{variable}_PLS2_pred_true_plot.jpg` and `.eps`)

### PLS2_testing.py
A program to test PLS2 models on spectra of samples with or without metadata. Reads variable names from filename of PLS2 model generated by `PLS2_modelling` or takes user input, but **if variable names inputted manually, they must be in the same order that the PLS2 model predicts them** for the predictions to be accurate. Also automatically pulls the relevant variable columns from the metadata. 
##### Input prompts
- PLS2 model filepath
- Test spectra filepath
- Test metadata filepath (optional)
##### Outputs 
- Predicted values of all variables (`PLS2_test_predictions_{variables).csv`)
- If have metadata, predicted vs. true values (`PLS2_test_pred_true_{variables}`) and scatter plots per variable (`{var}_PLS2_test_pred_true_plot.jpg` and `.eps`)

## Other files
### model_tools.py
Helper file that contains many of the functions and classes used by the above programs.

## Calibration model descriptions
Descriptions adapted from Dyar et al. (XXX)

### [Ordinary Least Squares (OLS)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
OLS is a standard linear regression method that chooses its coefficients by minimizing the sum of squares between the input data, **X**, and the independent variables, **y**. It assigns a weight to every channel in **X** and so provides interpretable values for the magnitude of the relationship between **X** and **y** at every point in the spectrum. OLS will always overfit to **X** because it is identical to the maximum likelihood estimator, and therefore can be useful as a ‘best’, albeit not useful, model to compare other linear calibration errors to.

### [Partial Least Squares (PLS)](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
PLS, also known as projection to latent structures, identifies fundamental relationships between **X** and **y** by transforming them into a lower-dimensional subspace and maximizing their covariance. It performs especially well in cases where the number of input channels (features), *p*, is significantly greater than the number of standards, N, and when the input data are highly correlated to each other. It is well suited to spectroscopy data, which are often large and collinear, where it has been used extensively for calibrations of other types of spectroscopy such as laser-induced breakdown  (Clegg et al., 2009; Boucher et al., 2015; Anderson et al., 2017), mid-infrared (Hecker et al., 2012; Pan et al., 2015; Breitenfeld et al., 2022) and Raman spectroscopy (de Groot et al., 2003; Huang et al., 2016; Breitenfeld et al., 2018) in geological problems and many other applications.

PLS identifies model coefficients by sequentially choosing directions, or components, where the covariance of **X** and **y** is maximized. It first shrinks **X** by projecting it from its *p*-dimensional space into a smaller k-dimensional vector space. Finally, OLS is used to regress the response on the components generated in the first step and minimize the residual sum of squared error.

### [the Least Absolute Shrinkage and Selection Operator (LASSO)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
LASSO regression is an OLS model that includes an ℓ1 shrinkage penalty on the model coefficients. This provides a sparse model (i.e., doesn’t use all channels in the spectrum) by shrinking some coefficients and setting most other coefficients to zero. LASSO assumes that a subset of input variables can be used to predict the outcome without reducing accuracy, which reduces a large and hard to interpret model to a smaller model containing only highly-correlated channels. 
To calculate its model coefficients, w, LASSO solves the following, where α is a constant that was optimized via cross-validation:

$$(min)┬w  1/(2n_"samples"  )||Xw-y||_2^2+α||w||_1$$

As α increases, fewer coefficients are chosen and the model becomes more robust to collinearity; thus when α = 0, LASSO is equivalent to OLS. This regularizer term smooths the model coefficients and prevents the model from overfitting the training data, while removing noisy features that may otherwise encumber the model.

### [Orthogonal Matching Pursuit (OMP)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html)
OMP approximates an optimized linear fit within a certain number of steps or by reaching a certain error tolerance using forward feature selection. It does so by iteratively selecting the channel in **X** most correlated to the current residual and adding this to a dictionary. At each step, the residual is recalculated with an orthogonal projection on the previously chosen channels. It is guaranteed to find the best fit within the constrained number of steps, or non-zero coefficients, ℓ0. The function is expressed as:

$$argmin┬w ||y-Xw||_2^2 " subject to "||w||_0≤l_0$$

when the number of non-zero coefficients is the constraint. The error tolerance-constrained function is as follows:

$$argmin┬w ||w||_0 " subject to "||y-Xw||_2^2≤"tol"$$

OMP therefore generates a sparse model like LASSO, but it differs in how the channels are chosen. Essentially, because OMP assigns coefficients sequentially, associations with the previously chosen channels are not seen by the model when it selects the next most correlated point.

### [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
ElasticNet is an OLS model that adds both ℓ1 and ℓ2 regularizations to the coefficients. The terms are added in an ℓ1 ratio, ρ, that combines the properties of LASSO and Ridge models; when ρ = 1, ElasticNet is equivalent to LASSO. ElasticNet exploits the feature selection capabilities of LASSO while adding the robustness of Ridge for data with high collinearity. It minimizes the following function:

$$(min)┬w  1/(2n_"samples"  )||Xw-y||_2^2+αρ||w||_1+(α(1-ρ))/2||w||_2^2$$

### [Ridge regression (Ridge)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
Ridge is very similar to LASSO in that it adds a regularizing term to OLS which penalizes the size of the model coefficients; however an ℓ2 penalty is used in this case:

$$(min)┬w ||Xw-y||_2^2+α||w||_2^2$$

The shrinkage parameter, α, was also optimized using cross-validation and similarly increases sparsity as its value increases. The difference in the regularization terms means that noisy features’ coefficients are shrunk but not set to zero with Ridge, so it does not have the feature selection capabilities of LASSO. 
However, it is more robust for identifying collinear variables because it will preserve all correlated features instead of selecting one randomly, as in LASSO.

### [Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
SVR finds the regression of best fit, the ‘hyperplane’ by maximizing the number of training samples with predicted values below a given error tolerance, ϵ,

$$|y_i-w_i x_i |≤ε$$

where w is the corresponding coefficient. The goal of SVR is thus to define a regression with the flattest ϵ ‘tube’, i.e., the narrowest error threshold, that contains most of the training predictions. In multivariate terms, the difference SVR has to other methods is that the goal is not to minimize the squared error, but rather to minimize the coefficient vector, w, while remaining below ϵ:

$$MIN 1/2 ‖w‖^2$$

Those samples with errors within the ϵ-tube are then ignored by the regression, while samples with residuals outside the threshold are called ‘support vectors’ and have the most influence over the shape of the tube. Therefore, SVR creates a more generalizable model by only using a subset of standards, the support vectors, to define the hyperplane.
SVR, like PCR, uses kernels to map **X** onto a higher dimension where hyperplanes that better reflect the data can be constructed. There are linear (**SVR-lin**) and nonlinear (2nd order polynomial, **SVR-py**) kernels to generate SVR models.

### [Principal Components Regression (PCR)](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html)
PCR first applies principal component analysis (PCA) to **X** that has been centered to zero and then OLS is fitted to the transformed **X**. PCA is an unsupervised method (i.e., does not take y into account) that works by identifying orthogonal vectors (the principal components) that explain most of the variance in **X** (defined as ‘eigenvectors’ of the covariance matrix of **X**). The effect of the first component (the one explaining most of the variance/has the largest ‘eigenvalue’) is then removed from the data, and the next principal component minimizes the variance of that result. This process is continued *p* times until all the variance in the data is explained by the principal components. Next, a dimensionality reduction step is often taken, where a subset of the first *k* components is selected to account for the majority of the variance in **X**, with *k* chosen by cross-validation. This step reduces sampling variation and overfitting, but there is a risk that components with low variance but high correlation to **y** may be dropped from the model. OLS is then performed on the k principal components. PCR’s dimensionality reduction is useful in cases where there are many collinear channels, but the presumption that **y** is related to the variance in **X** limits the scope of the model.

Sometimes the variance in **X** is not well explained by a linear vector **(PCR-lin)**. Instead, a kernel function can be used to map **X** onto a higher dimensional space where the eigenvectors can better fit the direction of highest variance. The choice of appropriate kernel function is dependent on the characteristics of **X**; a 2nd order polynomial kernel, K, can also be used **(PCR-py)**:

$$K(x,y)=(∑_i▒〖x_i y_i 〗)^2$$

### [Random Forest regressor (RF)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
RF is a nonlinear regression method that uses its ensemble properties to create robust predictions from many models based on the same training dataset. RF regression uses bootstrapping, a method for selecting random subsets of the training data, to create many randomized decision trees. The predictions from each tree are then averaged together to create a single prediction, which mitigates the random error from each model and defines the ‘ensemble’ methodology. Because the trees are made randomly, the ‘random state’ must be set for the results to be reproducible. A defined maximum number of features, *k*, (variables within **X**) are considered at each split of the decision trees.

### [Gradient Boosting regressor (GBR)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
GBR is another nonlinear ensemble method that uses decision trees to come to a final prediction. However, it is an additive method that begins with a small, simple decision tree that is not much better than a random guess. At each step, another weak tree is built with the goal of minimizing the squared error from the previous tree’s predictions and its estimator is added to the overall model. Over time, the regression becomes more accurate as it gets fine-tuned by each successive decision tree. This procedure is continued for n trees. Overfitting is avoided by using a ‘learning rate’, which scales the contribution of each tree’s result, e.g.,:

$$〖prediction〗_1=〖prediction〗_0+learning rate*〖estimator〗_1$$

### [k-Nearest Neighbors regressor (kNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
kNN is a nonlinear regression method that assigns an unknown prediction by averaging the values of the k training samples most similar to the unknown sample. This similarity is defined as the Euclidean distance to the unknown, and the neighbors’ values can either be averaged equally or weighted by the inverse of their distance. 
