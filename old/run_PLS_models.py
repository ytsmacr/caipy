import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from math import sqrt
import warnings
from os import path
import pickle
from statistics import mean
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression

'''
RUN_PLS_MODELS.PY by Cai Ytsma (ytsmacr@gmail.com)
Last updated 25/07/22

- takes training spectra (wave, pkey1, pkey2, etc.) and metadata (pkey, variable) files
- optionally also takes test spectra and metadata in same format
- performs CV automatically and chooses component w/ lowest RMSE-CV
- prints model components and RMSE-CV
- trains model to best component and prints RMSE-C, R2, adj. R2
- tests model on test data and prints RMSE-P, R2, adj R2
- exports model, model coefficients, and values listed above
'''

# TOOLS
def check_csv(filename):
    if filename[-4:] != '.csv':
        filename = filename + '.csv'
        
    return filename

def make_bool(val):
    if val == 'y':
        return True
    elif val == 'n':
        return False
    else:
        raise ValueError('Input needs to be either y or n')
        
def convert_spectra(spectra):
    cols = list(spectra.columns)
    if 'wave' not in cols:
        raise ValueError('Spectra file needs wavelength axis')
    axis = list(spectra['wave'].values)
    cols.remove('wave')
    
    spec_list = []
    for column in cols:
        spectrum = list(spectra[column])
        spec_list.append(spectrum)
    
    conv_spectra = np.array(spec_list)
    
    return axis, conv_spectra

def get_variable(meta):
    cols = list(meta.columns)
    if len(cols) > 2 or cols[0] != 'pkey':
        raise ValueError('Metadata file must only have two columns, with the first being pkey')
    var = cols[1]
    
    return var

# input information
data_folder = input('Folder path containing data: ')

X_train_path = path.join(data_folder, check_csv(input('Training spectra filename: ')))
y_train_path = path.join(data_folder, check_csv(input('Training metadata filename: ')))

do_test = make_bool(input('Do you have test data? (y/n) ').lower())
if do_test:
    X_test_path = path.join(data_folder, check_csv(input('Test spectra filename: ')))
    y_test_path = path.join(data_folder, check_csv(input('Test metadata filename: ')))

outpath = input('File path to export results: ')

# prep data
print('\nLoading data')
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

# check data in same order
train_check = list(X_train.columns[1:]) == list(y_train.pkey.values)
if not train_check:
    raise ValueError('Train spectra/metadata needs to be in same order')
train_names = list(y_train.pkey.values)

if do_test:
    test_check = list(X_train.columns[1:]) == list(y_train.pkey.values)
    if not test_check:
        raise ValueError('Test spectra/metadata needs to be in same order')
    test_names = list(y_test.pkey.values)


# format spectra
axis, X_train = convert_spectra(X_train)
if do_test:
    axis1, X_test = convert_spectra(X_test)
    if axis != axis1:
        raise ValueError('Test and train spectra need to have same wavelength axis')
        
# format metadata
var = get_variable(y_train)
y_train = y_train[var].values
if do_test:
    var1 = get_variable(y_test)
    if var != var1:
        raise ValueError('Train and test metadata variable must be the same')
    y_test = y_test[var].values
    
print('\nPerforming CV:')
# CROSS-VALIDATION
n_folds = 5
max_components = 30
component_range = np.arange(start=2, stop=max_components+1, step=1)

cv_dict = {}
for n_components in tqdm(component_range):
    # define model
    temp_pls = PLSRegression(n_components = n_components, scale=False)
    # run CV and get RMSE
    temp_rmsecv = (-cross_val_score(
        temp_pls, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error'
    )).mean()
    # add results to dictionary
    cv_dict[temp_rmsecv] = n_components

# select parameters of model with lowest rmsecv
rmsecv = min(list(cv_dict.keys()))
component = cv_dict[rmsecv]
model = PLSRegression(n_components = component, scale=False)
print(f'\n\tLowest RMSE-CV of {round(rmsecv,2)} obtained from {component}-component model')

# fit model
print(f'\nTraining {var} model:')
model.fit(X_train, y_train)
pickle.dump(model, open(f'{outpath}\\{var}_PLS_model.asc', 'wb'), protocol=0)

coef = pd.DataFrame({
    'wave':axis,
    'coef':model.coef_.flatten().tolist()
})
coef.to_csv(f'{outpath}\\{var}_PLS_coefs.csv', index=False)

train_preds = model.predict(X_train)
train_pred_true = pd.DataFrame({
    'pkey' : train_names,
    'actual' : y_train.flatten().tolist(),
    'pred' : train_preds.flatten().tolist()
})
train_pred_true.to_csv(f'{outpath}\\{var}_train_pred_true.csv', index=False)

rmsec = sqrt(mean_squared_error(train_pred_true.actual, train_pred_true.pred))
r2_train = model.score(X_train,y_train)
adj_r2_train = 1 - (1-r2_train)*(len(train_pred_true) - 1) / (len(train_pred_true) - (train_pred_true.shape[1] - 1) - 1)

print(f'\tRMSE-C: {round(rmsec,3)}    R2: {round(r2_train,3)}    Adjusted R2: {round(adj_r2_train,3)}')

# test model
if do_test:
    print(f'\nTesting {var} model:')
    test_preds = model.predict(X_test)
    test_pred_true = pd.DataFrame({
        'pkey' : test_names,
        'actual' : y_test.flatten().tolist(),
        'pred' : test_preds.flatten().tolist()
    })
    test_pred_true.to_csv(f'{outpath}\\{var}_test_pred_true.csv', index=False)

    rmsep = sqrt(mean_squared_error(test_pred_true.actual, test_pred_true.pred))
    r2_test = model.score(X_test,y_test)
    adj_r2_test = 1 - (1-r2_test)*(len(test_pred_true) - 1) / (len(test_pred_true) - (test_pred_true.shape[1] - 1) - 1)

    print(f'\tRMSE-P: {round(rmsep,3)}    R2: {round(r2_test,3)}    Adjusted R2: {round(adj_r2_test,3)}')


# output results
if do_test:
    results = pd.DataFrame({
        'variable':var,
        '# train':len(y_train),
        'RMSE-CV':rmsecv,
        '# components':component,
        'RMSE-C':rmsec,
        'R2 train':r2_train,
        'adj-R2 train':adj_r2_train,
        '# test':len(y_test),
        'RMSE-P':rmsep,
        'R2 test':r2_test,
        'adj-R2 test':adj_r2_test
    }, index=[0])
else:
    results = pd.DataFrame({
        'variable':var,
        '# train':len(y_train),
        'RMSE-CV':rmsecv,
        '# components':component,
        'RMSE-C':rmsec,
        'R2 train':r2_train,
        'adj-R2 train':adj_r2_train
    }, index=[0])

results.to_csv(f'{outpath}\\{var}_PLS_results.csv', index=False)
