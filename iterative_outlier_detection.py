import pandas as pd
import os
import numpy as np
from model_tools import *
from math import sqrt
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from scipy.stats import iqr
import argparse

'''
This module finds outliers by their poor predictions

by Cai Ytsma
last updated 25 August 2023

TO DO: 
[] Adjust threshold type to accept a list of multiple (like ML methods for automodelling)
[] Make ML parameters callable arguments (leaving static for now)
'''
###################################
'''
SETUP
'''
### DEFINE THINGS ###
var_prompt = 'Variable to identify outliers by (must be a metadata column name):'
om_prompt = 'Outlier detection method: either "per_spectrum" (default) or "per_sample" (requires column "Sample_Name")'
om_options = ['per_spectrum','per_sample']
tt_prompt = 'Procedure stopping threshold: either "max_iter" (default), "percent_samples", "rmse", or "percent_diff" (only works for "per_sample" outlier type)'
tt_options = ['percent_samples','rmse','max_iter','percent_diff']
tv_prompt = 'Procedure stopping threshold value (if using max_iter, leave blank for n_iter = N samples - 2)'
ml_prompt = '''Select a machine learning regression method:
    [1] Ordinary Least Squares (OLS)
    [2] Orthogonal Matching Pursuit (OMP)
    [3] the Least Absolute Shrinkage and Selection Operator (LASSO)
    [4] Ridge regression
    [5] ElasticNet
    [6] Partial Least Squares (PLS)
    [7] Principal Components Regression with linear PCA kernel (PCR-lin)
    [8] Principal Components Regression with 2nd degree polynomial PCA kernel (PCR-py)
    [9] Support Vector Regression with linear kernel (SVR-lin)
    [10] Support Vector Regression with 2nd degree polynomial kernel (SVR-py)
    [11] Random Forest regressor (RF)
    [12] Gradient Boosting regressor (GBR)
    [13] k-Nearest Neighbors regressor (kNN)'''
ml_dict = {
    1:'OLS',2:'OMP',3:'LASSO',4:'Ridge',5:'ElasticNet',6:'PLS',7:'PCR-lin',
    8:'PCR-py', 9:'SVR-lin',10:'SVR-py',11:'RF',12:'GBR',13:'kNN'
}

#maximum number of components for PLS
max_components = 15
# number of values to test for LASSO, Ridge, ElasticNet, SVR
num_params = 30
# polynomial degree for SVR and kernel PCR
poly_deg = 2
# maximum number of neighbors for kNN
max_neighbors = 40

### GET INPUTS ###
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--datafolder', type=str, default=None, help='Path of folder with data')
parser.add_argument('-o', '--outfolder', type=str, default=None, help='Path of folder to output results')
parser.add_argument('-s', '--spectra_name', type=str, default=None, help='Spectra filename')
parser.add_argument('-m', '--meta_name', type=str, default=None, help='Metadata filename')
parser.add_argument('-v', '--variable', type=str, default=None, help=var_prompt)
parser.add_argument('-ml', '--ml_method', type=int, default=None, choices=list(ml_dict.keys()), help=ml_prompt)
parser.add_argument('-om', '--outlier_method', type=str, default='per_spectrum', choices=om_options, help=om_prompt)
parser.add_argument('-tt', '--threshold_type', type=str, default='max_iter', choices=tt_options, help=tt_prompt)
parser.add_argument('-tv', '--threshold_value', default='default', help=tv_prompt)

### PULL VALUES ###
args=parser.parse_args()
data_folder = args.datafolder.replace("'","")
outfolder = args.outfolder.replace("'","")
spectra_path = args.spectra_name.replace("'","")
meta_path = args.meta_name.replace("'","")
variable = args.variable
ml_method = args.ml_method
outlier_method = args.outlier_method
threshold_type = args.threshold_type
threshold_value = args.threshold_value

### READ IN DATA ###
if data_folder is None:
    data_folder, all_files = get_data_folder()
else:
    all_files = os.listdir(data_folder)

if spectra_path is None:
    spectra_path = get_spectra_path(data_folder, all_files)
    spectra = pd.read_csv(spectra_path)
else:
    spectra = pd.read_csv(os.path.abspath(os.path.join(data_folder, spectra_path)))
    
if meta_path is None:
    meta_path = get_meta_path(data_folder, all_files)
    meta = pd.read_csv(meta_path)
else:
    meta = pd.read_csv(os.path.abspath(os.path.join(data_folder, meta_path)))

# check data formats
if spectra.columns[0] != 'wave':
    raise ValueError('First column of spectra file should be "wave": the x-axis values')
if meta.columns[0] != 'pkey':
    raise ValueError('First column of metadata file should be "pkey": the individual identifier of each spectrum')
    
# sort to make sure the same
if any(spectra.columns[1:] != list(meta.pkey)):
    print('Sorting to match')
    vals = list(meta.pkey)
    vals.sort()
    meta.sort_values('pkey', ignore_index=True, inplace=True)
    vals.insert(0,'wave')
    spectra = spectra[vals]

### ASSIGN VARIABLES ###
if outfolder is None:
    outfolder = get_out_folder()
if variable is None:
    variable = input(var_prompt)
    while variable not in meta.columns:
        print(f'Error: column {variable} not in metadata headers')
        variable = input(var_prompt)

if ml_method is None:
    ml_method = int(input(ml_prompt))
# convert to string
ml_method = ml_dict[ml_method]

if outlier_method is None:
    outlier_method = input(om_prompt)
    if outlier_method == '':
        outlier_method='per_spectrum'
        print('Using default of per_spectrum')
    while outlier_method not in om_options:
        print(f"Error: Outlier method {outlier_method} not in {', '.join(om_options)}")

if threshold_type is None:
    threshold_type = input(tt_prompt)
    if threshold_type == '':
        threshold_type='max_iter'
        print('Using default of max_iter')
    while threshold_type not in tt_options:
        print(f"Error: Threshold type {threshold_type} not in {', '.join(tt_options)}")
        threshold_type = input(tt_prompt)

# default for max_iter
def get_max_iter(method):
    if outlier_method == 'per_spectrum':
        print('Using default max_iter of N spectra - 2')
        return meta.pkey.nunique() - 2
    elif outlier_method == 'per_sample':
        print('Using default max_iter of N samples - 2')
        return meta.Sample_Name.nunique() - 2

# where using complete defaults, or specified max_iter with no threshold value
if (threshold_value == 'default')&(threshold_type == 'max_iter'):
    threshold_value = get_max_iter(outlier_method)
# where using max_iter, and provided threshold value
elif (threshold_value != 'default')&(threshold_type == 'max_iter'):
    threshold_value = int(threshold_value)
# where not using max_iter, and provided a value
elif (threshold_value != 'default')&(threshold_type != 'max_iter'):
    threshold_value = float(threshold_value)
# where no values provided
else:
    threshold_value = float(input(tv_prompt))

''' 
FUNCTIONS
'''
def get_model_results(method, formatted_data, variable, fold_col, test_fold=None, by_spectrum=False):
    '''
    Gets outcome of model for that current iteration

    Returns: train or train and test errors
    '''
    global max_components, num_params, poly_deg, max_neighbors
    # get data
    data_dict, min_samples = formatted_data.make_data_dict(variable, fold_col, test_fold)
    # initiate modelling class with data dictionary
    modelling = Model(data_dict, hide_progress=True)
    reg_cv_dict = {
    'PLS':{'func':modelling.run_PLS,
           'args':max_components},
    'LASSO':{'func':modelling.run_LASSO,
             'args':num_params},
    'Ridge':{'func':modelling.run_Ridge,
             'args':num_params},
    'ElasticNet':{'func':modelling.run_ElasticNet,
                  'args':num_params},
    'SVR-lin':{'func':modelling.run_SVR_linear,
               'args':num_params},
    'SVR-py':{'func':modelling.run_SVR_poly,
              'args':(num_params, poly_deg)},
    'PCR-lin':{'func':modelling.run_PCR_linear,
           'args':None},
    'PCR-py':{'func':modelling.run_PCR_poly,
             'args':poly_deg},
    'OMP':{'func':modelling.run_OMP,
           'args':None},
    'RF':{'func':modelling.run_RF,
           'args':None},
    'GBR':{'func':modelling.run_GBR,
           'args':None},
    'OLS':{'func':modelling.run_OLS,
           'args':None},
    'kNN':{'func':modelling.run_kNN,
           'args':max_neighbors}
    }
    # get RMSECV
    if type(reg_cv_dict[method]['args']) == int: # if has arguments...
        param, rmsecv, model = reg_cv_dict[method]['func'](reg_cv_dict[method]['args'])
    elif reg_cv_dict[method]['args']:
        param, rmsecv, model = reg_cv_dict[method]['func'](*reg_cv_dict[method]['args'])
    else:
        param, rmsecv, model = reg_cv_dict[method]['func']()
    # prep data for training
    if test_fold is not None:
        train_names, X_train, y_train, test_names, X_test, y_test = form.format_spectra_meta(variable, fold_col, test_fold)
    else:
        train_names, X_train, y_train = form.format_spectra_meta(variable, fold_col)              
    model.fit(X_train, y_train)
    # get RMSE-C
    train_preds = model.predict(X_train)
    train_pred_true = pd.DataFrame({
        'pkey' : train_names,
        'actual' : y_train.flatten().tolist(),
        'pred' : train_preds.flatten().tolist()
    })
    rmsec = sqrt(mean_squared_error(
        train_pred_true['actual'], 
        train_pred_true['pred'])
                )
    r2_train = model.score(X_train,y_train)
    if test_fold is not None:
        if by_spectrum is False:
            # get test value
            test_preds = model.predict(X_test)
            test_pred_true = pd.DataFrame({
                'pkey' : test_names,
                'actual' : y_test.flatten().tolist(),
                'pred' : test_preds.flatten().tolist()
            })
            rmsep = sqrt(mean_squared_error(
                test_pred_true['actual'], 
                test_pred_true['pred']
            ))
            # can't do percentage due to all the zeroes
            # test R2 wasn't helpful because only one point
            return rmsecv, rmsec, r2_train, rmsep
        else:
            test_pred = model.predict(X_test)
            diff = y_test - test_pred
            return rmsecv, rmsec, r2_train, diff
    else:
        return rmsecv, rmsec, r2_train

def identify_outlier(df, train_col, test_col):
    '''
    Identifies the outlier
    '''
    # by spectrum - easy
    if test_col == 'test_diff':
        outlier = df.sort_values(test_col, ascending=False, ignore_index=True)['pkey'][0]
    # by sample - more complex
    elif test_col == 'RMSEP':
        # easiest case - one has the lowest train and highest test
        worst_train = df.sort_values(train_col, ignore_index=True)['Sample_Name'][0]
        worst_test = df.sort_values(test_col, ascending=False, ignore_index=True)['Sample_Name'][0]
        if worst_train == worst_test:
            outlier = worst_train
        # then, use the IQR to help
        else:
            # want low RMSE
            train_RMSE_lower = np.percentile(df[train_col], 25)
            # want high RMSEP
            test_RMSE_upper = np.percentile(df[test_col], 75)
        
            # samples outside the custom IQR
            potential_outliers = df[
                (result_df[train_col] < train_RMSE_lower)&
                (result_df[test_col] > test_RMSE_upper)
            ].reset_index(drop=True).copy()
            
            # if more than one, choose that with the lowest train RMSE
            if len(potential_outliers)>1:
                outlier = potential_outliers.sort_values(train_col, ignore_index=True)['Sample_Name'][0]
            elif len(potential_outliers)==0:
                print('Halting procedure: No obvious outliers found')
                # this implies that the values are stable?
                data_plot(train_col, test_col, df)
                outlier = False
            else:
                outlier = potential_outliers['Sample_Name'][0]
    return outlier

'''
RUN PROCEDURE
'''
print(f'Performing iterative outlier removal {outlier_method} for variable {variable} with a {threshold_type} stopping threshold of {threshold_value}, using a {ml_method} regression')

# universal variables
train_col = 'avg_train_RMSE'
to_continue = True
count=1
outlier_list = []
rmse_list = []
diff_list = []
per_diff_list = []

### SPECIFICS BY METHOD ###
if outlier_method == 'per_sample':
    # define things
    test_col = 'RMSEP'
    id_col = 'Sample_Name'
    n_samples = meta.Sample_Name.nunique()
    # check
    if id_col not in meta.columns:
        raise ValueError('Metadata file must have column "Sample_Name" to perform a per sample outlier detection method')
    # assign a fold per sample
    fold_df = pd.DataFrame(meta[id_col].unique()).reset_index()
    fold_df.columns = ['Folds',id_col]
    meta = meta.merge(fold_df, how='left')

elif outlier_method == 'per_spectrum':
    # define things
    test_col = 'test_diff'
    id_col = 'pkey'
    n_samples = len(meta)
    # assign a fold per spectrum
    meta = meta.reset_index().rename(columns={'index':'Folds'})

### GENERAL TO DEFINE ###
if threshold_type == 'percent_samples':
    # depending whether input as whole number or already converted to decimal
    if threshold_value < 1:
        max_n_outliers = int(threshold_value * n_samples)
    else:
        threshold_value = threshold_value/100
        max_n_outliers = int(threshold_value * n_samples)

# fold to sample key
sample_dict = dict(zip(meta['Folds'], meta[id_col]))
# sample to variable value key
var_dict = dict(zip(meta[id_col], meta[variable]))
        
# copy them to avoid writing over
spectra_ = spectra.copy()
meta_ = meta.copy()

### BEGIN ###
while to_continue is True:
    # prep data formatting
    form = Format(spectra_, meta_)
    fold_col = form.get_fold_col(variable)

    # get original RMSEC
    if count==1:
        rmsecv, rmsec, r2_train = get_model_results(ml_method, form, variable, fold_col)
        rmse = round((rmsecv + rmsec)/2,2)
        outlier_list.append('no outliers removed')
        rmse_list.append(rmse)
        diff_list.append(np.nan)
        per_diff_list.append(np.nan)

    print(f'Finding outlier #{count}')
    result_data = []
    for test_fold in tqdm(list(meta_.Folds.unique()), desc='Sample', leave=False):
        sample = sample_dict[test_fold]
        # make model
        if outlier_method == 'per_sample':
            rmsecv, rmsec, r2_train, rmsep = get_model_results(ml_method, 
                                                               form, 
                                                               variable, 
                                                               fold_col, 
                                                               test_fold)
            results = [test_fold, sample, rmsecv, rmsec, r2_train, rmsep]
        elif outlier_method == 'per_spectrum':
            rmsecv, rmsec, r2_train, diff = get_model_results(ml_method, form, variable, fold_col, test_fold, by_spectrum=True)
            results = [test_fold, sample, rmsecv, rmsec, r2_train, diff]
        result_data.append(results)
    # make df
    result_cols = ['Fold',id_col,'RMSECV','RMSEC','R2_train',test_col]
    result_df = pd.DataFrame(result_data, columns=result_cols)
    result_df['avg_train_RMSE'] = result_df[['RMSEC','RMSECV']].mean(axis=1)
    
    # find outlier
    outlier = identify_outlier(result_df, train_col, test_col)   
    to_continue = False if outlier is False else to_continue # exit if needed
    if outlier is not False:
        outlier_list.append(outlier)
        # save that value as the train RMSE to match to
        current_rmse = result_df[result_df[id_col]==outlier][train_col].values[0]
        rmse_list.append(current_rmse)

        # find difference between this and the last value
        last_rmse = rmse_list[-2]
        diff = current_rmse - last_rmse
        per_diff = round((diff/last_rmse)*100,1)
        diff_list.append(diff)
        per_diff_list.append(per_diff)
    
        # remove outlier and prep for next iteration
        meta = meta[meta[id_col]!=outlier].reset_index(drop=True)
        cols = list(meta.pkey)
        cols.insert(0,'wave')
        spectra = spectra[cols]
    
        count+=1

        # see if passes a breakage threshold
        if threshold_type == 'max_iter':
            if count >= threshold_value+1:
                print('Halting procedure: Reached maximum number of iterations')
                to_continue = False
        if threshold_type == 'rmse':
            if current_rmse <= threshold_value:
                if to_continue is True:
                    print(f'Halting procedure: Current training RMSE of {round(current_rmse,2)} below threshold of {threshold_value}')
                to_continue = False
        if threshold_type == 'percent_diff':
            if per_diff <= threshold_value:
                if to_continue is True:
                    print(f'Halting procedure: Training RMSE after last outlier removal has a difference to the previous model below the threshold of {threshold_value}%')
                to_continue = False
        if threshold_type == 'percent_samples':
            n_outliers = count-1
            # don't subtract 1 because checking to see if a new outlier would push it over the threshold
            if n_outliers == max_n_outliers:
                if to_continue is True:
                    print(f'Halting procedure: Reached the maximum of {n_outliers} outliers, which was defined as ~{round(threshold_value*100)}% of all samples')
                to_continue = False
    
# collate results
outlier_result_df = pd.DataFrame({
    'outlier':outlier_list,
    'avg_train_RMSE':rmse_list,
    'difference':diff_list,
    'percent_difference':per_diff_list
})
# export
outfile = os.path.join(outfolder, f'iterative_outlier_results_{variable.replace("/","-")}_{outlier_method}_{ml_method}_threshold_{round(threshold_value,1)}_{threshold_type}.csv')
outlier_result_df.to_csv(outfile, index=False)