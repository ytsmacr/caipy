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
Last updated 24 October 2022

Apply .asc sklearn model to input data. Returns .csv of predicted values.
Optionally include metadata file for test samples to generate:
    - RMSE-P, R2, adjusted R2
    - predicted vs. true scatter plot

Spectra file column format:
'wave' (wavelength axis), {sample_1} (spectral intensities), {sample_2}, etc.

Metadata file format:
'pkey' (sample names), {variable} (values to be predicted), optionally 'Sample Name' or 'Sample_Name'
'''

# GET FILE INFORMATION
# model
model_prompt = 'Model file path: (e.g. C:\Documents\SiO2_model.asc) '
model_file = check_asc(input(model_prompt))
while not os.path.exists(model_file):
    print(f'Error: path {model_file} does not exist')
    model_file = check_asc(input(model_prompt))

print('\n***REMINDER***\nTest data should be processed identically to how training data were processed\n')

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
    comps_prompt = 'Test metadata file path: (e.g. C:\Documents\metadata.csv) '
    meta_file = check_csv(input(comps_prompt))
    while not os.path.exists(meta_file):
        print(f'Error: path {meta_file} does not exist')
        meta_file = check_csv(input(comps_prompt))
        
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

# format data
X_test = convert_spectra(spectra)
# get predictions
test_pred = model.predict(X_test)

# get variable and model information from filename if possible
model_name = model_file.split('\\')[-1]
if re.match('.+_.+_model\.asc$', model_name):
    var = model_name.split('_')[0]
else:
    var = input('Error: Could not extract variable name from model filename.\nWhat is the relevant variable? ')

if re.match('.+_.+_model.asc$', model_name):
    method = model_name.split('_')[1]
else:
    method = input('Error: Could not extract regression method from model filename.\nWhat is the relevant method? ')
    
# make df of results
actual_col = f'{var}_actual'
pred_col = f'{var}_pred'

pred_df = pd.DataFrame({
    'pkey':spectra.columns[1:],
    pred_col:list(test_pred.flatten())
})

if not have_comps:
    pred_df.to_csv(f'{outpath}\\{var}_{method}_predictions.csv', index=False)
    print('Exported predicted values')

else:
    # see if var in meta file
    count = 0
    while var not in meta.columns:
        if count >= 1:
            var_cols = ', '.join([col for col in meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)])
            print('Metadata columns to choose from: ', var_cols)
        var = input(f'Error: {var} not a metadata column. What is the relevant variable? ')
        count += 1

    # add actual values and rearrange/format
    for option in ['Sample_Name', 'Sample Name']:
        if option in meta.columns:
            cols_to_add = ['pkey', var, option]
            final_cols = ['pkey', option, pred_col, actual_col]

    pred_true = pred_df.merge(meta[cols_to_add], how='left', on='pkey')
    pred_true.rename(columns={var:actual_col}, inplace=True)
    pred_true = pred_true[final_cols]
    
    # RMSE-P
    rmsep = sqrt(mean_squared_error(pred_true[actual_col],
                                    pred_true[pred_col]))
    # R2
    r2 = r2_score(pred_true[actual_col],
                  pred_true[pred_col])
    # adjusted r2
    adj_r2 = 1 - (1-r2)*(len(pred_true) - 1) / (len(pred_true) - (pred_true.shape[1] - 1) - 1)
    print(f'\n\tRMSE-P: {round(rmsep,3)}    R2: {round(r2,3)}    Adjusted R2: {round(adj_r2,3)}\n')
    
    # PLOT
    Plot.pred_true(df = pred_true,
                   var = var, 
                   method = method, 
                   type = 'test',
                   rmse = rmsep,
                   adj_r2 = adj_r2,
                   path = outpath)
    pred_true.to_csv(f'{outpath}\\{var}_{method}_pred_true.csv', index=False)
    print('Exported predicted vs. true values and plot')