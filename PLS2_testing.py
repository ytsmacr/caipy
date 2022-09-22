import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import re

from model_tools import check_csv, check_asc, make_bool, convert_spectra, Plot

'''
by Cai Ytsma (cai@caiconsulting.co.uk)
Last updated 22 September 2022

Test .asc sklearn PLS2 model on input data. Returns .csv of predicted values.
Optionally include metadata file for test samples to generate:
    - RMSE-P, R2, adjusted R2
    - predicted vs. true scatter plots

Spectra file column format:
'wave' (wavelength axis), {sample_1} (spectral intensities), {sample_2}, etc.

Metadata file format:
'pkey' (sample names), optionally 'Sample_Name', 
{variable1}, {variable2}, etc. (values to be predicted; same as in original PLS2 model)
'''

# GET FILE INFORMATION
# model
model_prompt = 'Model file path: (e.g. C:\Documents\PLS2_model_SiO2_TiO2.asc) '
model_file = check_asc(input(model_prompt))
while not os.path.exists(model_file):
    print(f'Error: path {model_file} does not exist')
    model_file = check_asc(input(model_prompt))

print('\n***REMINDER***\nTest spectra should be processed identically to how training data were processed\n')

# spectra
spectra_prompt = 'Test spectra file path: (e.g. C:\Documents\spectra.csv) '
spectra_file = check_csv(input(spectra_prompt))
while not os.path.exists(spectra_file):
    print(f'Error: path {spectra_file} does not exist')
    spectra_file = check_csv(input(spectra_prompt))

# have compositions for test samples?
comps_prompt = 'Do you have compositions for these samples (y/n): '
have_comps = make_bool(input(comps_prompt).lower())
while have_comps == 'error':
        print('Error: Input needs to be either y or n')
        have_comps = make_bool(input(comps_prompt).lower())
# if so, get comps
if have_comps:
    print('\n***REMINDER***\nTest metadata should contain all variables for which the PLS2 model predicts\n')
    test_prompt = 'Test metadata file path: (e.g. C:\Documents\metadata.csv) '
    meta_file = check_csv(input(test_prompt))
    while not os.path.exists(meta_file):
        print(f'Error: path {meta_file} does not exist')
        meta_file = check_csv(input(test_prompt))
        
# folder to export results to
out_prompt = 'File path to export results: '
outpath = input(out_prompt)
while not os.path.exists(outpath):
    print(f'Error: path {outpath} does not exist\n')
    outpath = input(out_prompt)
    
# load files
model = pickle.load(open(model_file, 'rb'))
spectra = pd.read_csv(spectra_file)
if have_comps:
    meta = pd.read_csv(meta_file)
    # check data in same order
    check = list(spectra.columns[1:]) == list(meta['pkey'].values)
    if not check:
        raise ValueError('Spectra and metadata samples need to be in same order')
        
# get variable information from filename if possible
model_name = model_file.split('\\')[-1]
# remove suffix
model_name = model_name[:-4]
# get variables
var_list = [v for v in model_name.split('_') if v not in ['PLS2', 'model']]

# get variable information from filename if possible
model_name = model_file.split('\\')[-1]
pattern = '^PLS2_model_((.)+_)+.+\.asc$'
if re.match(pattern, model_name):
    # remove suffix
    model_name = model_name[:-4]
    # get variables
    var_list = [v for v in model_name.split('_') if v not in ['PLS2', 'model']]
else:
    var_list = input('Error: Could not extract variable names from model filename.\nWhat are the relevant variables? (separated by a space *IN THE ORDER THE MODEL PREDICTS THEM*):').split()
    
# format data
X_test = convert_spectra(spectra)

# get predictions
test_pred = model.predict(X_test)
pred_df = pd.DataFrame(test_pred)
pred_df.columns = [f'{var}_pred' for var in var_list]
pred_df.insert(0,'pkey',spectra.columns[1:])

if not have_comps:
    pred_df.to_csv(f"{outpath}\\PLS2_test_predictions_{'_'.join(var_list)}.csv", index=False)
    print('Exported predicted values')
    
# predicted vs true
if have_comps:
    
    # check if have variables
    cols = [c for c in meta.columns if c in var_list+['pkey', 'Sample_Name']]
    missing_var = []
    for var in var_list:
        if var not in cols:
            missing_var.append(var)
    if len(missing_var) != 0:
        raise ValueError(f"Variable column(s) missing from test metadata: {', '.join(missing_var)}")
    
    # format meta
    actual_df = meta[cols]
    actual_df.columns = [f'{c}_actual' if c in var_list else c for c in actual_df.columns]
    # combine
    pred_true = actual_df.merge(pred_df)
    # export
    pred_true.to_csv(f"{outpath}\\PLS2_test_pred_true_{'_'.join(var_list)}.csv", index=False)
    
    # get results for each variable
    rmsep_list = []
    r2_list = []
    adj_r2_list = []
    
    for var in var_list:

        actual = pred_true[f'{var}_actual']
        pred = pred_true[f'{var}_pred']

        rmsep = sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        adj_r2 = 1 - (1-r2)*(len(pred_true) - 1) / (len(pred_true) - (pred_true.shape[1] - 1) - 1)

        rmsep_list.append(rmsep)
        r2_list.append(r2)
        adj_r2_list.append(adj_r2)
        
        # PLOT
        Plot.pred_true(df = pred_true,
                       var = var, 
                       method = 'PLS2', 
                       type = 'test',
                       rmse = rmsep,
                       adj_r2 = adj_r2,
                       path = outpath)

    test_results = pd.DataFrame({
        'Variable' : var_list,
        'RMSE-P' : rmsep_list,
        'R2' : r2_list,
        'Adjusted R2' : adj_r2_list
    })
    # export
    test_results.to_csv(f"{outpath}\\PLS2_test_results_{'_'.join(var_list)}.csv", index=False)
    print('Exported predicted vs. true values and plot')