import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from math import sqrt
import warnings
import os
import pickle
import re
import time
import argparse

from model_tools import *

'''
by Cai Ytsma (cai@caiconsulting.co.uk)
Last updated 26 October 2023

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
    [8] Principal Components Regression with 2nd degree polynomial PCA kernel (PCR-py)
    [9] Support Vector Regression with linear kernel (SVR-lin)
    [10] Support Vector Regression with 2nd degree polynomial kernel (SVR-py)
    [11] Random Forest regressor (RF)
    [12] Gradient Boosting regressor (GBR)
    [13] k-Nearest Neighbors regressor (kNN)
...or a combination, separated by a space or comma?: '''
std_prompt = 'Should all variables follow the same modelling procedure? (set test fold, model type(s)) (y/n): '
test_prompt = 'Do you want to use one of the folds as a test set? Otherwise all data used for training (y/n): '
test_prompt2 = 'Which fold should be the test fold? '

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

# from arguments
parser = argparse.ArgumentParser()

parser.add_argument('-f', '--datafolder', type=str, default=None, help='Path of folder with data')
parser.add_argument('-o', '--outpath', type=str, default=None, help='Path of folder to output results')
parser.add_argument('-s', '--spectra_name', type=str, default=None, help='Spectra filename')
parser.add_argument('-m', '--meta_name', type=str, default=None, help='Metadata filename')
parser.add_argument('-std', '--standard', action='store_true', help='Follow a standard procedure for each variable (bool)')
parser.add_argument('-dt', '--do_test', action='store_true', help='Holds a fold out as test data')
parser.add_argument('-mt', '--method', type=str, default=None, help=f'Number corresponding to method selection from: {method_prompt}')
parser.add_argument('-tf', '--test_fold', type=int, default=None, help='Integer of fold to be used for testing')
parser.add_argument('-hp', '--hide_progress', action='store_true', help='Hides progress bars')
parser.add_argument('-mc', '--max_components', type=int, default=None, help='Sets the maximum PLS components')
parser.add_argument('-np', '--num_params', type=int, default=None, help='Sets the number of values to test for LASSO, Ridge, ElasticNet, SVR')
parser.add_argument('-pd', '--poly_deg', type=int, default=None, help='Sets the polynomial degree for SVR and kernel PCR')
parser.add_argument('-mn', '--max_neighbors', type=int, default=None, help='Sets the maximum number of neighbors for kNN')

args=parser.parse_args()
data_folder = args.datafolder.replace("'","")
outpath = args.outpath.replace("'","")
spectra_path = args.spectra_name.replace("'","")
meta_path = args.meta_name.replace("'","")
standard = args.standard
do_test = args.do_test
method_type = args.method
if method_type is not None:
    method_type = method_type.replace("'","")
test_fold = args.test_fold
hide_progress = args.hide_progress
max_components_ = args.max_components
num_params_ = args.num_params
poly_deg_ = args.poly_deg
max_neighbors_ = args.max_neighbors

# from inputs
if data_folder is None:
    data_folder, all_files = get_data_folder()
else:
    all_files = os.listdir(data_folder)

if outpath is None:
    outpath = get_out_folder()

# make folder if it doesn't already exist
if not os.path.exists(outpath):
    os.mkdir(outpath)    
    
# read in data
if spectra_path is None:
    spectra_path = get_spectra_path(data_folder, all_files)
    spectra = pd.read_csv(spectra_path)
else:
    spectra = pd.read_csv(os.path.join(data_folder, spectra_path))
    
if meta_path is None:
    meta_path = get_meta_path(data_folder, all_files)
    meta = pd.read_csv(meta_path)
else:
    meta = pd.read_csv(os.path.join(data_folder, meta_path))

# show the progress bars and results of CV
if hide_progress is None:
    hide_progress = False

if max_components_ is None:
    max_components_ = 30
if num_params_ is None:
    num_params_ = 30
if poly_deg_ is None:
    poly_deg_ = 2
if max_neighbors_ is None:
    max_neighbors_ = 40
    
#----------------#
# PREP PROCEDURE #
#----------------#
# wavelength axis to store for later
axis = list(spectra['wave'].values)

# check data in same order
check = list(spectra.columns[1:]) == list(meta['pkey'].values)
if not check:
    raise ValueError('Spectra and metadata samples need to be in same order')
    
# make class instance for formatting
form = Format(spectra, meta)
    
# extract variables to be run
var_to_run = [col for col in meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)]
all_var = ', '.join(var_to_run)
print('Identified variable(s) to model:', all_var)

# check if run same procedure for all var
if standard is None:
    standard = False
    if len(var_to_run) > 1:
        standard = make_bool(input(std_prompt).lower())
        while standard == 'error':
            print('Error: Input needs to be either y or n')
            standard = make_bool(input(std_prompt).lower())

# if so, get parameters
if standard:
    if do_test is None:
        do_test = make_bool(input(test_prompt).lower())

        while do_test == 'error':
            print('Error: Input needs to be either y or n')
            do_test = make_bool(input(test_prompt).lower())
    if do_test:
        if test_fold is None:
            test_fold = int(input(test_prompt2))
    
    while True:
        if method_type is None:
            method_type = input(method_prompt)
        div = ',' if ',' in method_type else ' '
        method_types = [int(x) for x in method_type.split(div)]
        if set(method_types).issubset(set(method_dict.keys())):
            if 0 in method_types:
                methods_torun = method_dict[0]
                print("\nPerforming", ', '.join([str(x) for x in methods_torun]), "regressions")
                break
            if len(method_types)>1:    
                methods_torun = []
                for m in method_types:
                    methods_torun = methods_torun + method_dict[m]
                    methods_torun = list(set(methods_torun))
                print("\nPerforming", ', '.join([str(x) for x in methods_torun]), "regressions")
            elif len(method_types) == 1:
                methods_torun = method_dict[method_types[0]]
                print(f'\nPerforming {methods_torun[0]} regression')
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
    max_components = max_components_
    # number of values to test for LASSO, Ridge, ElasticNet, SVR
    num_params = num_params_
    # polynomial degree for SVR and kernel PCR
    poly_deg = poly_deg_
    # maximum number of neighbors for kNN
    max_neighbors = max_neighbors_
    
    print(f'\nRunning for {var}')
    fold_col = form.get_fold_col(var)
    
    # ask for testing and method choice if not standardized procedure
    if not standard:
        if do_test is None:
            do_test = make_bool(input(test_prompt).lower())
            if do_test:
                if test_fold is None:
                    test_fold = int(input(test_prompt2))

    # get data in the correct format for CV
    if do_test:
        all_folds = ', '.join([str(x) for x in meta[fold_col].unique() if x != -1])
        while test_fold not in meta[fold_col].unique():
            print(f'{test_fold} not in list of available folds: {all_folds}')
            test_fold = int(input(test_prompt2))
        data_dict, min_samples = form.make_data_dict(var, fold_col, test_fold)
    else:
        data_dict, min_samples = form.make_data_dict(var, fold_col)
        
    # update parameters if larger than min samples
    print('min samples:', min_samples)
    
    max_components = len(spectra) if max_components > len(spectra) else max_components
    #num_params = min_samples if num_params > min_samples else num_params
    max_neighbors = min_samples if max_neighbors > min_samples else max_neighbors
        
    if not standard:
        while True:
            if method_type is None:
                method_type = input(method_prompt)
            div = ',' if ',' in method_type else ' '
            method_types = [int(x) for x in method_type.split(div)]
            if set(method_types).issubset(set(method_dict.keys())):
                if 0 in method_types:
                    methods_torun = method_dict[0]
                    print("\nPerforming", ', '.join([str(x) for x in methods_torun]), "regressions")
                    break
                if len(method_types)>1:    
                    methods_torun = []
                    for m in method_types:
                        methods_torun = methods_torun + method_dict[m]
                        methods_torun = list(set(methods_torun))
                    print("\nPerforming", ', '.join([str(x) for x in methods_torun]), "regressions")
                elif len(method_types) == 1:
                    methods_torun = method_dict[method_types[0]]
                    print(f'\nPerforming {methods_torun[0]} regression')
                break
            else:    
                print(f"Error: Input must be one of {all_methods}")

    # initiate modelling class with data dictionary
    modelling = Model(data_dict, hide_progress)
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
        print(f'\nPerforming CV for {method}')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(reg_cv_dict[method]['args']) == int: # if has arguments...
                param, rmsecv, model = reg_cv_dict[method]['func'](reg_cv_dict[method]['args'])
            elif reg_cv_dict[method]['args']:
                 param, rmsecv, model = reg_cv_dict[method]['func'](*reg_cv_dict[method]['args'])
            else:
                param, rmsecv, model = reg_cv_dict[method]['func']()
            
        # get data in format for full model
        print(f'\nTraining model')
        if do_test:
            train_names, X_train, y_train, test_names, X_test, y_test = form.format_spectra_meta(var, fold_col, test_fold)
        else:
            train_names, X_train, y_train = form.format_spectra_meta(var, fold_col)

        # fit training model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        pickle.dump(model, open(os.path.join(outpath, f'{var}_{method}_model.asc'), 'wb'), protocol=0)

        # MODEL PARAMETERS
        if method in non_linear_methods:
            print(f'{method} is non-linear so does not generate coefficients or an intercept')
            intercept = 'NA'
        
        else:       
            # special cases here
            if method in ['SVR-lin']:
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

            coef.to_csv(os.path.join(outpath, f'{var}_{method}_coefs.csv'), index=False)

            # plot
            Plot.coeffs(df = coef,
                        spectrum = X_train[0],
                        var = var,
                        method = method,
                        path = outpath)

        # PREDICTIONS
        actual_col = f'{var}_actual'
        pred_col = f'{var}_pred'
        
        train_preds = model.predict(X_train)
        train_pred_true = pd.DataFrame({
            'pkey' : train_names,
            actual_col : y_train.flatten().tolist(),
            pred_col : train_preds.flatten().tolist()
        })
        train_pred_true.to_csv(os.path.join(outpath, f'{var}_{method}_train_pred_true.csv'), index=False)

        rmsec = sqrt(mean_squared_error(train_pred_true[actual_col], train_pred_true[pred_col]))
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
            print(f'\nTesting model')
            # TEST PREDICTIONS
            test_preds = model.predict(X_test)
            test_pred_true = pd.DataFrame({
                'pkey' : test_names,
                actual_col : y_test.flatten().tolist(),
                pred_col : test_preds.flatten().tolist()
            })
            test_pred_true.to_csv(f'{outpath}\\{var}_{method}_test_pred_true.csv', index=False)

            rmsep = sqrt(mean_squared_error(test_pred_true[actual_col], test_pred_true[pred_col]))
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
    print(f'\n{var} took {round(sub_end-sub_start,1)} seconds to run')
    
#----------------#
# EXPORT RESULTS #
#----------------#

# if only had training
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
    
    results.to_csv(os.path.join(outpath,'modelling_train_results.csv'), index=False)

# if had training and testing    
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

    results.to_csv(os.path.join(outpath,'modelling_train_test_results.csv'), index=False)

if len(var_to_run) > 1:
    main_end = time.time()
    print(f'\nAll variables took {round((main_end-main_start)/60,1)} minutes to run')
