#general packages
from functools import partial
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from statistics import mean, median
from tqdm import tqdm
import os
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import statistics
plt.set_loglevel('error')
from utilities import modelUtilities
# preprocessing
from tools.airPLS import airPLS
from tools.spectres import spectres
from sklearn.preprocessing import normalize

# modelling
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, OrthogonalMatchingPursuit
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from globals import Globals
import multiprocessing.pool
from joblib import Parallel, delayed

# CLASS OF PARALLEL REGRESSION FUNCTIONS (PLS, ELASTICNET, LASSO)
def get_first_local_minimum(li):
    min = li[0]
    for i in li[1:]:
        if i < min:
            min = i   
        elif i > min:
            return min
        elif i == min:
            continue
    # in case the last point is the lowest
    return min
def run_CV( model):
    rmsep_list = []
    for fold in list(Globals.data_dict.keys()):
        # get data
        X_train = Globals.data_dict[fold]['train_spectra']
        X_test = Globals.data_dict[fold]['test_spectra']
        y_train = Globals.data_dict[fold]['train_metadata']
        y_test = Globals.data_dict[fold]['test_metadata']

        # run model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
       # print(preds)
        test_df = pd.DataFrame({
            'actual': y_test.flatten().tolist(),
            'pred' : preds.flatten().tolist()
        })
        rmsep = sqrt(mean_squared_error(test_df.actual, test_df.pred))
        rmsep_list.append(rmsep)
    rmsecv = mean(rmsep_list) 
    return rmsecv

#function being parallelized?
def base_PLS(n_components):
    print("COMPONENETS:", n_components)
    # define model
    model = PLSRegression(n_components = n_components, scale=False)
    print(model)
     # run CV and get RMSE
    temp_rmsecv =run_CV( model)
    # add results to dictionary
   # cv_dict[temp_rmsecv] = n_components
    return model, temp_rmsecv, n_components

def PLS_parallel(max_components):
    # with Pool(1) as p:
    #     p.map(base_PLS, [2, 3, 4])
    cv_dict = {}
    out = Parallel(n_jobs=-1, verbose = 1, prefer = "threads")(delayed(base_PLS)(n_components = components)  for components in range(2,max_components +1))
    for components, values in enumerate(out):
        model, temp_rmsecv, n_components = values[0], values[1], values[2]
        cv_dict[temp_rmsecv] = n_components
    rmsecv = get_first_local_minimum(list(cv_dict.keys()))
    component = cv_dict[rmsecv]
    model = PLSRegression(n_components = component, scale=False)

    if Globals.hide_progress is False:
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from {component}-component model')
    return component, rmsecv, model

def base_LASSO(alpha):
    model = Lasso(alpha=alpha)
    temp_rmsecv = run_CV(model)
    return model, temp_rmsecv, alpha
    

def LASSO_parallel(num_alphas):
    alpha_range = np.logspace(-10, 1, num_alphas)
    cv_dict = dict()
    out = Parallel(n_jobs=-1, verbose = 1, prefer = "threads")(delayed(base_LASSO)(alpha = alpha)  for alpha in tqdm(alpha_range, desc='alpha value', disable=Globals.hide_progress))
    for alphas, values in enumerate(out):
        model, temp_rmsecv, alpha = values[0], values[1], values[2]
        cv_dict[temp_rmsecv] = alpha
    rmsecv = get_first_local_minimum(list(cv_dict.keys()))
    alpha = cv_dict[rmsecv]
    model = Lasso(alpha=alpha)
    if Globals.hide_progress is False:
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(alpha,5)}')
    return alpha, rmsecv, model
    

def base_Ridge(alpha):
    model = Ridge(alpha=alpha)
    temp_rmsecv = run_CV(model)
    return model, temp_rmsecv, alpha

def Ridge_parallel(num_alphas):
    alpha_range = np.logspace(-10, 1, num_alphas)
    cv_dict = dict()
    out = Parallel(n_jobs=-1, verbose = 1, prefer = "threads")(delayed(base_Ridge)(alpha = alpha)  for alpha in tqdm(alpha_range, desc='alpha value', disable=Globals.hide_progress))
    for alphas, values in enumerate(out):
        model, temp_rmsecv, alpha = values[0], values[1], values[2]
        cv_dict[temp_rmsecv] = alpha
    rmsecv = get_first_local_minimum(list(cv_dict.keys()))
    alpha = cv_dict[rmsecv]
    model = Ridge(alpha=alpha)
    if Globals.hide_progress is False:
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(alpha,5)}')
    return alpha, rmsecv, model


#takes ratio and alpha from below function ratio_and_alpha_runner_ElasticNet and runs the model, rerturning the rmsecv and alpha and ratio values to be read in by ElasticNet_parallel at the end
def base_ElasticNet(alpha, ratio):
    #declare global pool variable
    model = ElasticNet(alpha=alpha, l1_ratio=ratio)
    temp_rmsecv = modelUtilities.run_CV( model)
    return temp_rmsecv, alpha, ratio

#calls base_elasticnet, which runs the model for a SINGLE ratio over the range of alpha values
#equivalent to inner loop that goes over the alpha values for each ratio in original implementation of ElasticNet
def ratio_and_alpha_runner_ElasticNet(ratio, num_alphas):
    #declare global pool variable
    global pool
    #use num_alphas to create alpha range
    alpha_range = np.logspace(-7, 1, num_alphas)
    #equivalent to second loop that takes a given ratio and the range of numn alphas and makes a job out of it
    inputs = [[alpha, ratio] for alpha in tqdm(alpha_range, desc='alpha value', leave=False, disable=Globals.hide_progress)]
    return pool.starmap(base_ElasticNet, inputs)

#ElasticNet in parallel using pool multiprocessing, num_alphas same argument input as original implemenation
#calls ratio_and_alpha_runner_ElasticNet for each ratio value it iterates through (equivalent to outer loop in nested loop implementation of original function)
'''
Running:
python3 models.py -f /Users/sananayasna/Winternship/caipy2.0/caipy2.0/caipy/tests/test_data -o /Users/sanjanayasna/Winternship/caipy2.0/caipy2.0/caipy/tests/results -m test_meta_stratified.csv -s test_spectra.csv -std True -dt True -mt 5 -loq n 
'''
def ElasticNet_parallel( num_alphas):
    #declare global pool variable
    global pool
    count = 0
    # suggested by documentation to skew to lasso
    ratio_range = [.1, .5, .7, .9, .95, .99, 1]
   # ratio_range = [.1, .5]
    # slightly raise min because takes longer
    alpha_range = np.logspace(-7, 1, num_alphas)
    count = 0
    total_jobs = len(ratio_range) * len(alpha_range)
    #total number of jobs to run via global pool variable: alpha range x ratio range
    cv_dict = dict()
    pool = multiprocessing.pool.ThreadPool(total_jobs)
    inputs = [[ratio, num_alphas] for ratio in tqdm(ratio_range, desc='L1 ratio', leave=False, disable=Globals.hide_progress)]

    #equivalent to first loop that loops over ratios
    results = pool.starmap(ratio_and_alpha_runner_ElasticNet, inputs)

    #close pool
    pool.close()

    #from results array, which is num of ratios by num of alphas, get the min return rmsecv
    # print(len(results)) #number of ratios
    # print(len(results[0])) #number of alphas
    # print(len(results[0][1])) #number of values returned, and it should be a triple as base_ElasticNet returns 3 values

    #dummy values
    min_rmsecv = 99999999
    alpha_val = 0
    ratio_val = 0

    #extract the minimum rmsecv from pool job result returns
    for i in range(len(results)): #iterate over number of ratios
        for j in range(len(results[0])): #iterate over number of alphas
            #extract rmsecv value returned
            extracted_rmsecv = results[i][j][0]
            #if new min, fill in alpha_val and ratio vals
            if extracted_rmsecv < min_rmsecv:
                min_rmsecv = extracted_rmsecv
                alpha_val = results[i][j][1]
                ratio_val = results[i][j][2]
            
    print("Min rmsecv" ,min_rmsecv)
    print("alpha val", alpha_val)
    print("L + ratio", ratio_val)
 
    #make model from this:
    model = ElasticNet(alpha=alpha_val, l1_ratio=ratio_val)
    if Globals.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(min_rmsecv,2)} obtained from model with an alpha of {round(alpha_val,5)} and an l1_ratio of {ratio_val}')
    param = f'alpha={alpha_val} l1_ratio={ratio_val}'
    return param, min_rmsecv, model 

    