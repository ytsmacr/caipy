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

def get_variable1(meta):
    cols = list(meta.columns)
    if cols[0] != 'pkey':
        raise ValueError('First metadata column must be pkey')
    var1 = cols[1]
    
    return var1

def get_variable2(meta):
    cols = list(meta.columns)
    var2 = cols[2]
    
    return var2
    
def get_variable3(meta):
    cols = list(meta.columns)
    var3 = cols[3]
    
    return var3

# input information
data_folder = input('Folder path containing data: ')

X_train_path = path.join(data_folder, check_csv(input('Training spectra filename: ')))
y_train_path = path.join(data_folder, check_csv(input('Training metadata filename: ')))

# prep data
print('\nLoading data')
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)

# check data in same order
train_check = list(X_train.columns[1:]) == list(y_train.pkey.values)
if not train_check:
    raise ValueError('Train spectra/metadata needs to be in same order')
train_names = list(y_train.pkey.values) 
    
# format spectra
axis, X_train = convert_spectra(X_train)
        
# format metadata
var1 = get_variable1(y_train)
var2 = get_variable2(y_train)
var3 = get_variable3(y_train)
y_train = y_train[[var1, var2, var3]].values #edited [var1].values

print('\ny-train')
print(y_train)

  
#running model
print('\nPerforming CV:')

n_folds = 16
max_components = 20
component_range = np.arange(start=2, stop=max_components+1, step=1)

cv_dict = {}
for n_components in tqdm(component_range):
    # define model
    temp_pls = PLSRegression(n_components = n_components, scale=False)
    
    MSE = (-cross_val_score(temp_pls, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')).mean()
    temp_rmsecv = MSE
    
    # add results to dictionary
    cv_dict[temp_rmsecv] = n_components

# select parameters of model with lowest rmsecv
msecv = min(list(cv_dict.keys()))
rmsecv = sqrt(msecv)
component = cv_dict[msecv]
model = PLSRegression(n_components = component, scale=False)
print(f'\n\tLowest RMSE-CV of {round(rmsecv,2)} obtained from {component}-component model')

