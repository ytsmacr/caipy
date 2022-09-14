import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from math import sqrt
import warnings
import os
import pickle
import re
import time

from model_tools import *

'''
by Cai Ytsma (cai@caiconsulting.co.uk)
Last updated 14 September 2022

Train spectral calibration standards with PLS and/or LASSO modelling. 
Optionally use one fold of standards as test set.

Spectra file column format:
'wave' (wavelength axis), {sample_1} (spectral intensities), {sample_2}, etc.

Metadata file format:
'pkey' (sample names), 'Folds' OR '{variable}_Folds' (folds to split data), {variable} (values to be predicted), etc.

'Folds' will be used as fold column if no {variable}_Folds column is specified.

OUTPUT:
- Model file
    {variable}_{model}_model.asc

- Model-predicted versus true values and XY plots 
    {variable}_{model}_{train/test}_pred_true.csv
    {variable}_{model}_{train/test}_pred_true_plot.jpg
    {variable}_{model}_{train/test}_pred_true_plot.eps

- Model coefficient values and plot over example spectrum
    {variable}_{model}_coefs.csv
    {variable}_{model}_coefs_plot.jpg
    {variable}_{model}_coefs_plot.eps
    
- Overall results
    # train/test, model parameters, RMSEs, R2s
'''

#-------------------------------------------------------------#
#                     DEFINED VARIABLES                       #
#-------------------------------------------------------------#

# PROMPTS
method_prompt = '''Do you want to run:
    [0] *ALL MODELS*
    [1] Ordinary Least Squares (OLS)
    [2] Orthogonal Matching Pursuit (OMP)
    [3] the Least Absolute Shrinkage and Selection Operator (LASSO)
    [4] Ridge regression
    [5] ElasticNet
    [6] Partial Least Squares (PLS)
    [7] Principal Components Regression with linear PCA kernel (PCR-lin)
    [8] Principal Components Regression with linear PCA kernel (PCR-py)
    [9] Support Vector Regression with 2nd degree polynomial kernel (SVR-lin)
    [10] Support Vector Regression with 2nd degree polynomial kernel (SVR-py)
    [11] Random Forest regressor (RF)
    [12] Gradient Boosting regressor (GBR)
    [13] k-Nearest Neighbors regressor (kNN)
...or a combination, separated by a space or comma?: '''
std_prompt = 'Should all variables follow the same modelling procedure? (set test fold, model type(s)) (y/n): '
test_prompt = 'Do you want to use one of the folds as a test set? Otherwise all data used for training (y/n): '

# REGRESSION STUFF
method_dict = {
    0:['OLS','OMP','LASSO','Ridge','ElasticNet','PLS','PCR-lin','PCR-py','SVR-lin','SVR-py','RF','GBR','kNN'],
    1:['OLS'],
    2:['OMP'],
    3:['LASSO'],
    4:['Ridge'],
    5:['ElasticNet'],
    6:['PLS'],
    7:['PCR-lin'],
    8:['PCR-py'],
    9:['SVR-lin'],
    10:['SVR-py'],
    11:['RF'],
    12:['GBR'],
    13:['kNN']
}

all_methods = method_dict.keys()
non_linear_methods = ['SVR-py', 'PCR-lin', 'PCR-py', 'RF', 'GBR', 'kNN']
#-------------------------------------------------------------#

#-------------------#
# INPUT INFORMATION #
#-------------------#
# data folder
data_folder = input('Folder path containing data: ')
while not os.path.exists(data_folder):
    print(f'Error: path {data_folder} does not exist\n')
    data_folder = input('Folder path containing data: ')
    
all_files = os.listdir(data_folder)

# spectra
spectra_file = check_csv(input('Spectra filename: '))
while spectra_file not in all_files:
    print(f'Error: file {spectra_file} not in data folder\n')
    spectra_file = check_csv(input('Spectra filename: '))
spectra_path = os.path.join(data_folder, spectra_file)

# metadata
meta_file = check_csv(input('Metadata filename: '))
while meta_file not in all_files:
    print(f'Error: file {meta_file} not in data folder\n')
    meta_file = check_csv(input('Metadata filename: '))
meta_path = os.path.join(data_folder, meta_file)

# folder to export results to
outpath = input('File path to export results: ')
while not os.path.exists(outpath):
    print(f'Error: path {outpath} does not exist\n')
    outpath = input('File path to export results: ')
    
#----------------#
# PREP PROCEDURE #
#----------------#
# read in data
print('\nLoading data')
spectra = pd.read_csv(spectra_path)
axis = list(spectra['wave'].values)
meta = pd.read_csv(meta_path)

# make class instance for formatting
form = Format(spectra, meta)

# check data in same order
check = list(spectra.columns[1:]) == list(meta['pkey'].values)
if not check:
    raise ValueError('Spectra and metadata samples need to be in same order')
    
# extract variables to be run
var_to_run = [col for col in meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)]
all_var = ', '.join(var_to_run)
print('Identified variable(s) to model:', all_var)

# check if run same procedure for all var
standard = False
if len(var_to_run) > 1:
    standard = make_bool(input(std_prompt).lower())
    while standard == 'error':
        print('Error: Input needs to be either y or n')
        standard = make_bool(input(std_prompt).lower())

# if so, get parameters
if standard:
    do_test = make_bool(input(test_prompt).lower())
    
    while do_test == 'error':
        print('Error: Input needs to be either y or n')
        do_test = make_bool(input(test_prompt).lower())
    if do_test:
        test_fold = int(input(f'Which fold should be the test fold? '))
    
    while True:
        method_type = input(method_prompt)
        method_types = [int(x) for x in re.findall('\d', str(method_type))]
        if set(method_types).issubset(set(method_dict.keys())):
            if len(method_types)>1:    
                methods_torun = []
                for m in method_types:
                    methods_torun = methods_torun + method_dict[m]
                    methods_torun = list(set(methods_torun))
            elif len(method_types) == 1:
                methods_torun = method_dict[method_types[0]]
            break
        else:    
            print(f"Error: Input must be one of {all_methods}")
            
#---------------#
# RUN PROCEDURE #
#---------------#
# prep lists for results df
n_train_list = []
rmsecv_list = []
param_list = []
rmsec_list = []
intercept_list = []
r2_train_list = []
adj_r2_train_list = []
n_test_list = []
rmsep_list = []
r2_test_list = []
adj_r2_test_list = []
test_fold_list = []
method_list = []
var_list = []

# get elapsed time rather than tqdm
main_start = time.time()

for var in var_to_run:
    
    # RESET MODEL PARAMETERS
    #maximum number of components for PLS
    max_components = 30
    # number of values to test for LASSO, Ridge, ElasticNet, SVR
    num_params = 30
    # polynomial degree for SVR and kernel PCR
    poly_deg = 2
    # maximum number of neighbors for kNN
    max_neighbors = 40
    
    print(f'\nRunning for {var}')
    fold_col = form.get_fold_col(var)
    
    # ask for testing and method choice if not standardized procedure
    if not standard:
        do_test = make_bool(input(test_prompt).lower())
        if do_test:
            test_fold = int(input(f'Which fold should be the test fold? '))

    # get data in the correct format for CV
    if do_test:
        all_folds = ', '.join([str(x) for x in meta[fold_col].unique() if x != -1])
        while test_fold not in meta[fold_col].unique():
            print(f'{test_fold} not in list of available folds: {all_folds}')
            test_fold = int(input(f'Which fold should be the test fold? '))
        data_dict, min_samples = form.make_data_dict(var, fold_col, test_fold)
    else:
        data_dict, min_samples = form.make_data_dict(var, fold_col)
        
    # update parameters if larger than min samples
    max_components = min_samples if max_components > min_samples else max_components
    num_params = min_samples if num_params > min_samples else num_params
    max_neighbors = min_samples if max_neighbors > min_samples else max_neighbors
        
    if not standard:
        while True:
            method_type = input(method_prompt)
            method_types = [int(x) for x in re.findall('\d+', str(method_type))]
            if set(method_types).issubset(set(method_dict.keys())):
                if len(method_types)>1:    
                    methods_torun = []
                    for m in method_types:
                        methods_torun = methods_torun + method_dict[m]
                        methods_torun = list(set(methods_torun))
                        print("\Performing", ', '.join([str(x) for x in methods_torun]), "regressions")
                        break
                elif len(method_types) == 1:
                    methods_torun = method_dict[method_types[0]]
                    print(f'\nPerforming {methods_torun[0]} regression')
                    break
            else:    
                print(f"Error: Input(s) must be among {all_methods}")

    # initiate modelling class with data dictionary
    modelling = Model(data_dict)
    # functions and arguments per method
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
    
    # check length of time for all methods
    sub_start = time.time()
    
    for method in methods_torun:
        
        # optimize models with CV
        print(f'\nPerforming CV for {method}:')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(reg_cv_dict[method]['args']) == int: # if has arguments...
                param, rmsecv, model = reg_cv_dict[method]['func'](reg_cv_dict[method]['args'])
            elif reg_cv_dict[method]['args']:
                 param, rmsecv, model = reg_cv_dict[method]['func'](*reg_cv_dict[method]['args'])
            else:
                param, rmsecv, model = reg_cv_dict[method]['func']()
            
        # get data in format for full model
        print(f'\nTraining model:')
        if do_test:
            train_names, X_train, y_train, test_names, X_test, y_test = form.format_spectra_meta(var, fold_col, test_fold)
        else:
            train_names, X_train, y_train = form.format_spectra_meta(var, fold_col)

        # fit training model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        pickle.dump(model, open(f'{outpath}\\{var}_{method}_model.asc', 'wb'), protocol=0)

        # MODEL PARAMETERS
        if method in non_linear_methods:
            print(f'{method} is non-linear so does not generate coefficients or an intercept')
            intercept = 'NA'
        
        else:       
            # special cases here
            if method == 'SVR-lin':
                coef_list = list(model.coef_[0])
                intercept = model.intercept_[0]
            else:
                coef_list = list(model.coef_)
                intercept = model.intercept_
            
            coef = pd.DataFrame({
                'wave':axis,
                'coef':coef_list
            })
            if not pd.api.types.is_numeric_dtype(coef['coef']):
                coef['coef'] = [x[0] for x in coef.coef]

            coef.to_csv(f'{outpath}\\{var}_{method}_coefs.csv', index=False)

            # plot
            Plot.coeffs(df = coef,
                        spectrum = X_train[0],
                        var = var,
                        method = method,
                        path = outpath)

        # PREDICTIONS
        train_preds = model.predict(X_train)
        train_pred_true = pd.DataFrame({
            'pkey' : train_names,
            'actual' : y_train.flatten().tolist(),
            'pred' : train_preds.flatten().tolist()
        })
        train_pred_true.to_csv(f'{outpath}\\{var}_{method}_train_pred_true.csv', index=False)

        rmsec = sqrt(mean_squared_error(train_pred_true.actual, train_pred_true.pred))
        r2_train = model.score(X_train,y_train)
        adj_r2_train = 1 - (1-r2_train)*(len(train_pred_true) - 1) / (len(train_pred_true) - (train_pred_true.shape[1] - 1) - 1)
        
        # plot
        Plot.pred_true(df = train_pred_true, 
                       var = var, 
                       method = method, 
                       type = 'train', 
                       rmse = rmsec, 
                       adj_r2 = adj_r2_train, 
                       path = outpath)

        print(f'\tRMSE-C: {round(rmsec,3)}    R2: {round(r2_train,3)}    Adjusted R2: {round(adj_r2_train,3)}')

        # add data to lists
        var_list.append(var)
        n_train_list.append(len(y_train))
        method_list.append(method)
        rmsecv_list.append(rmsecv)
        intercept_list.append(intercept)
        param_list.append(param)
        rmsec_list.append(rmsec)
        r2_train_list.append(r2_train)
        adj_r2_train_list.append(adj_r2_train)

        # optional testing
        if do_test:
            print(f'\nTesting model:')
            # TEST PREDICTIONS
            test_preds = model.predict(X_test)
            test_pred_true = pd.DataFrame({
                'pkey' : test_names,
                'actual' : y_test.flatten().tolist(),
                'pred' : test_preds.flatten().tolist()
            })
            test_pred_true.to_csv(f'{outpath}\\{var}_{method}_test_pred_true.csv', index=False)

            rmsep = sqrt(mean_squared_error(test_pred_true.actual, test_pred_true.pred))
            r2_test = model.score(X_test,y_test)
            adj_r2_test = 1 - (1-r2_test)*(len(test_pred_true) - 1) / (len(test_pred_true) - (test_pred_true.shape[1] - 1) - 1)
            
            # Plot
            Plot.pred_true(df = test_pred_true, 
                           var = var, 
                           method = method, 
                           type = 'test', 
                           rmse = rmsep, 
                           adj_r2 = adj_r2_test, 
                           path = outpath)
            
            print(f'\tRMSE-P: {round(rmsep,3)}    R2: {round(r2_test,3)}    Adjusted R2: {round(adj_r2_test,3)}')

            test_fold_list.append(test_fold)
            n_test_list.append(len(y_test))
            rmsep_list.append(rmsep)
            r2_test_list.append(r2_test)
            adj_r2_test_list.append(adj_r2_test) 
        else:
            test_fold_list.append('NA')
            n_test_list.append('NA')
            rmsep_list.append('NA')
            r2_test_list.append('NA')
            adj_r2_test_list.append('NA')  
            
    # report elapsed time for variable
    sub_end = time.time()
    print(f'{var} took {round(sub_end-sub_start,1)} seconds to run')
    
#----------------#
# EXPORT RESULTS #
#----------------#
if (set(test_fold_list) == {'NA'}) and (set(n_test_list) == {'NA'}) and(set(rmsep_list) == {'NA'}) and(set(r2_test_list) == {'NA'}) and(set(adj_r2_test_list) == {'NA'}):
    results = pd.DataFrame({
        'variable':var_list,
        '# train':n_train_list,
        'model type':method_list,
        'RMSE-CV':rmsecv_list,
        'model parameter':param_list,
        'model intercept':intercept_list,
        'RMSE-C':rmsec_list,
        'R2 train':r2_train_list,
        'adj-R2 train':adj_r2_train_list
    })
    
    results.to_csv(f'{outpath}\\modelling_train_results.csv', index=False)
else:
    results = pd.DataFrame({
        'variable':var_list,
        '# train':n_train_list,
        'model type':method_list,
        'RMSE-CV':rmsecv_list,
        'model parameter':param_list,
        'model intercept':intercept_list,
        'RMSE-C':rmsec_list,
        'R2 train':r2_train_list,
        'adj-R2 train':adj_r2_train_list,
        'test fold':test_fold_list,
        '# test':n_test_list,
        'RMSE-P':rmsep_list,
        'R2 test':r2_test_list,
        'adj-R2 test':adj_r2_test_list
    })

    results.to_csv(f'{outpath}\\modelling_train_test_results.csv', index=False)

if len(var_to_run) > 1:
    main_end = time.time()
    print(f'All variables took {round((main_end-main_start)/60,1)} minutes to run')