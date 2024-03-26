import sys
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
import os
import pickle
import re
import time
import argparse
#class feed args to global.py
from globals import *
from utilities import foldUtilities, generalUtilities, dataLoadingUtilities,  modelUtilities, plotUtilities
from analyze import Analyze
import parallel as parallel 

#__________________________________________________________________
# MODELSETUP CLASS:
# Makes dictionary of necessary model mdethods,
# gets most representative test folds if appliable
# cross validate function
# applies model(s) from model_dict made
#__________________________________________________________________
class modelSetup:
    #overal argument parser that calls utilties functions as aid
    def handle_args(self, sysargs = None):
        method_prompt = '''Select which of following to run:
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
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--datafolder', type=str, default=None, help='Path of folder with data')
        parser.add_argument('-o', '--outpath', type=str, default=None, help='Path of folder to output results')
        parser.add_argument('-s', '--spectra_name', type=str, default=None, help='Spectra filename')
        parser.add_argument('-m', '--meta_name', type=str, default=None, help='Metadata filename')
        parser.add_argument('-std', '--standard', type=bool, default=None, help='Follow a standard procedure for each variable (bool)')
        parser.add_argument('-dt', '--do_test', type=bool, default=None, help='Hold a fold out as test data (bool)')
        parser.add_argument('-mt', '--method', type=str, default=None, help=f'Number corresponding to method selection from: {method_prompt}')
        parser.add_argument('-mc', '--max_components', type=int, default=30, help='Sets the maximum PLS components')
        parser.add_argument('-np', '--num_params', type=int, default=30, help='Sets the number of values to test for LASSO, Ridge, ElasticNet, SVR')
        parser.add_argument('-pd', '--poly_deg', type=int, default=2, help='Sets the polynomial degree for SVR and kernel PCR')
        parser.add_argument('-mn', '--max_neighbors', type=int, default=40, help='Sets the maximum number of neighbors for kNN')
        parser.add_argument('-loq','--do_loq', type=str, default=None, help='Post modelling function: Calculate LOQ of each sample for each regression model, and then recalculate rmse and r2 statistics from the train and test prediction data. y for yes, n for no: ')
        parser.add_argument('-sen', '--sensitivity', type=float, default=None, help='Float value for the sensitivty value of the calibration instrument')
        parser.add_argument('-pa', '--parallel', type=str, default=None, help='Run regression models in parallel?y for yes, n for no: ')
        args, unknown=parser.parse_known_args(sysargs)
        data_folder = args.datafolder
        #fill out globals if args parser args supplied
        if data_folder is not None:
            data_folder = data_folder.replace("'","")
            Globals.data_folder = data_folder
        outpath = args.outpath
        if outpath is not None:
            outpath = outpath.replace("'","")
            Globals.outpath = outpath
        spectra_path = args.spectra_name
        if spectra_path is not None:
            spectra_path = spectra_path.replace("'","")
            Globals.spectra_path = spectra_path
        meta_path = args.meta_name
        if meta_path is not None:
            meta_path = meta_path.replace("'","")
            Globals.meta_path = meta_path
        Globals.standard = args.standard
        Globals.do_test = args.do_test
        Globals.parallel = args.parallel
        if args.parallel is not None:
            Globals.parallel = generalUtilities.make_bool(args.parallel)
        #get meta and spectra data loaded 
        dataLoadingUtilities.directories(Globals.data_folder, Globals.outpath, Globals.spectra_path, Globals.meta_path)
        #get model parameters updated if supplied
        Globals.max_components_ = args.max_components
        Globals.num_params_ = args.num_params
        Globals.poly_deg_ = args.poly_deg
        Globals.max_neighbors_ = args.max_neighbors
        Globals.sensitivity = args.sensitivity
        
        if args.do_loq is not None:
            if str.lower(args.do_loq) == "y":
                Globals.do_loq = True
            elif str.lower(args.do_loq) == "n":
                Globals.do_loq = False
            else: raise ValueError("invalid input for argument do_loq/loq")
        if Globals.sensitivity is not None and Globals.do_loq is False:
            raise AttributeError("Sensitivity value provided yet do_loq is false. Contradictory input")
        #get list of variables to run, if none: 
        if len(Globals.var_to_run) == 0:
            Globals.var_to_run = [col for col in Globals.meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)]
            print("variables to run: ", Globals.var_to_run)
        dataLoadingUtilities.handle_standard()
        dataLoadingUtilities.models_args()
       
    #sets fold dictionaries (unique folds per variable, selected test fold per variable)
    #creates temp_meta dictionary too
    def set_folds(self):
        #if not do_test, then just set non_test_folds_per_var immediately, as there is no test_fold specific dictionary
        Globals.var_to_run = [col for col in Globals.meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)]
        if not Globals.do_test:
            #test folds dict is empty, fill out the non test fold sby figuring out unique folds per var
            
            for var in Globals.var_to_run:
                Globals.test_fold_per_var[var] = None
                assert var in Globals.meta.columns
                # get the folds
                all_folds = list(Globals.meta[f'{var}_Folds'].unique())
                # ignore the samples that aren't to be modelled
                if -1 in all_folds:
                    all_folds.remove(-1)
                all_folds.sort()
                #not necessary, delete
                # #only one fold, so test fold can't be done (was this meant for get most representative folds?)
                # if (len(all_folds) == 1):
                #     Globals.non_test_folds_per_var[var] = all_folds
                #     Globals.test_fold_per_var = []
                #     continue 
                Globals.non_test_folds_per_var[var] = all_folds
            #temp meta for train/test df formation is meta itself
            Globals.temp_meta = Globals.meta.copy()
            print("No folds selected exclusively for testing")
            print("Unique folds by sample:", Globals.non_test_folds_per_var)
        #if do_test, set both non_test_folds_per_var and test_fold_per_var (handled in foldUtilities.et_most_representative_folds function)
        else:
            foldUtilities.get_most_representative_folds()

#trains/tests/predicts with models, and exports the analyze script if do_loq is true. 
class trainModels:
    def apply_models_non_parallel(self):
        # OPEN RESULTS FILE
        outfile = open(os.path.join(Globals.outpath,'modelling_results.csv'), 'w')
        # enter header based on whether or not loq will be called
        if Globals.do_loq:
            if Globals.do_test: #add the extra test statistics from the analyze script to the headers, otherwise just the train stastistics
                outfile.writelines('variable,n_train,model_type,rmsecv,model_params,model_intercept,rmsec,r2_train,adj_r2_train,test_fold,n_test,rmsep,r2_test,adj_r2_test,median_conc_train,loq,n_train_above_loq,median_conc_train_above_loq,rmsec_above_loq,r2_train_above_loq,adj_r2_train_above_loq,median_conc_test,n_test_above_loq,median_conc_test_above_loq,rmsep_above_loq,r2_test_above_loq,adj_r2_test_above_loq\n')
            #loq statistics minus those involving test, since this is without do_test
            else: outfile.writelines('variable,n_train,model_type,rmsecv,model_params,model_intercept,rmsec,r2_train,adj_r2_train,test_fold,n_test,rmsep,r2_test,adj_r2_test,median_conc_train,loq,n_train_above_loq,median_conc_train_above_loq,rmsec_above_loq,r2_train_above_loq,adj_r2_train_above_loq\n')
        else: 
            outfile.writelines('variable,n_traPin,model_type,rmsecv,model_params,model_intercept,rmsec,r2_train,adj_r2_train,test_fold,n_test,rmsep,r2_test,adj_r2_test\n')
            
        # get elapsed time rather than tqdm
        
        main_start = time.time()
        for var in Globals.var_to_run:
            # RESET MODEL PARAMETERS
            #maximum number of components for PLS
            max_components_ = Globals.max_components_
            # number of values to test for LASSO, Ridge, ElasticNet, SVR
            num_params_ = Globals.num_params_
            # polynomial degree for SVR and kernel PCR
            poly_deg_ = Globals.poly_deg_
            # maximum number of neighbors for kNN
            max_neighbors_ = Globals.max_neighbors_

            #Get sample test fold
            print(f'\nRunning for {var}')
            fold_col = foldUtilities.get_fold_col(var)
            if Globals.test_fold_per_var[var] is None:
                test_fold = None
            else: test_fold = Globals.test_fold_per_var[var]
            Globals.data_dict, min_samples = foldUtilities.make_data_dict(var, fold_col, test_fold)

            # print(var, Globals.non_test_folds_per_var)
            # Test folds selection dictionary: {'SiO2': 1, 'TiO2': 1, 'Al2O3': 1}
            # Unique folds by sample: {'SiO2': [2, 3, 4, 5], 'TiO2': [2, 3, 4, 5], 'Al2O3': [2, 3, 4, 5]}
            # SiO2 {'SiO2': [2, 3, 4, 5], 'TiO2': [2, 3, 4, 5], 'Al2O3': [2, 3, 4, 5]}
            # TiO2 {'SiO2': [2, 3, 4, 5], 'TiO2': [2, 3, 4, 5], 'Al2O3': [2, 3, 4, 5]}
            # Al2O3 {'SiO2': [2, 3, 4, 5], 'TiO2': [2, 3, 4, 5], 'Al2O3': [2, 3, 4, 5]}


            # update parameters if larger than min samples
            print('min samples:', min_samples)
            # max_components and max_neighbors local to run iteration, so Global default doesn't change
            max_components_ = min(len(Globals.spectra), Globals.max_components_) 
            num_params_ = min(Globals.num_params_, min_samples)
            #num_params = min_samples if num_params > min_samples else num_params
            max_neighbors_ = min(min_samples, Globals.max_neighbors_) 

             # functions and arguments per method
            reg_cv_dict = {
                'OLS':{'func':modelUtilities.run_OLS,
                    'args':None},
                'OMP':{'func':modelUtilities.run_OMP,
                    'args':None},
                'LASSO':{'func': modelUtilities.run_LASSO,
                        'args':num_params_},
                'Ridge':{'func':modelUtilities.run_Ridge,
                        'args':num_params_},
                'ElasticNet':{'func':modelUtilities.run_ElasticNet,
                            'args':num_params_},
                'PLS':{'func': modelUtilities.run_PLS,       #modelUtilities.run_PLS,
                    'args':max_components_},
                'PCR-lin':{'func':modelUtilities.run_PCR_linear,
                    'args':None},
                'PCR-py':{'func':modelUtilities.run_PCR_poly,
                        'args': poly_deg_},
                'SVR-lin':{'func':modelUtilities.run_SVR_linear,
                        'args':num_params_},
                'SVR-py':{'func':modelUtilities.run_SVR_poly,
                        'args':(num_params_, poly_deg_)},
                'RF':{'func':modelUtilities.run_RF,
                    'args':None},
                'GBR':{'func':modelUtilities.run_GBR,
                    'args':None},
                'kNN':{'func':modelUtilities.run_kNN,
                    'args':max_neighbors_}
            }

            # check length of time for all methods
            #methods_dict wasn't initialized yet
            
            methods_torun = list(Globals.methods_dict[var])
            names = (list(reg_cv_dict.keys()))
            non_linear_methods = ['SVR-py', 'PCR-lin', 'PCR-py', 'RF', 'GBR', 'kNN']
            #axis values for Globals set. You'd have to run prepare script to resample spectra for revised axis
            Globals.axis = list(Globals.spectra['wave'].values)
            #start timer for method
            sub_start = time.time()
            #run selected models for this variable/sample:
            for method in methods_torun:
                # optimize models with CV
                #THIS IS BUILDING/TRAINING MODEL
                print(f'\nPerforming CV for {names[method]}')
                type(reg_cv_dict[names[method]]['args']) 
                #feed in arguments for each method selected
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if type(reg_cv_dict[names[method]]['args'])  == int: # if has arguments...
                        param, rmsecv, model = reg_cv_dict[names[method]]['func'](reg_cv_dict[names[method]]['args'])
                    elif reg_cv_dict[names[method]]['args']:
                        param, rmsecv, model = reg_cv_dict[names[method]]['func'](*reg_cv_dict[names[method]]['args'])
                    else:
                        param, rmsecv, model = reg_cv_dict[names[method]]['func']()
                       # get data in format for full model
                print(f'\nTraining model')
                if Globals.do_test:
                    train_names, X_train, y_train, test_names, X_test, y_test = modelUtilities.format_spectra_meta(var, fold_col, test_fold)
                else:
                    train_names, X_train, y_train =  modelUtilities.format_spectra_meta(var, fold_col)
                #make the method variable the regression method name as opposed to just hte corresponding number of tyhe method
                method = names[method]
                # fit training model, dump asc in outpath destination
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                pickle.dump(model, open(os.path.join(Globals.outpath, f'{var}_{method}_model.asc'), 'wb'), protocol=0)
                # MODEL PARAMETERS
                if method in non_linear_methods:
                    print(f'{method} is non-linear so does not generate coefficients or an intercept')
                    intercept = 'NA'
                else:       
                    # special cases here
                    # the structure for pls may change in teh future
                    # DO TEST TO SEE STRUCTURE ITS IN TO THEN GO WITH THAT
                    if method in ['SVR-lin','PLS']:
                        coef_list = list(model.coef_[0])
                        intercept = model.intercept_[0]               
                    else:
                        coef_list = list(model.coef_)
                        intercept = model.intercept_
                    
                    coef = pd.DataFrame({
                        'wave':Globals.axis,
                        'coef':coef_list
                    })
                    if not pd.api.types.is_numeric_dtype(coef['coef']):
                        coef['coef'] = [x[0] for x in coef.coef]

                    coef.to_csv(os.path.join(Globals.outpath, f'{var}_{method}_coefs.csv'), index=False)

                    # plot
                    plotUtilities.coeffs(df = coef,
                                spectrum = X_train[0],
                                var = var,
                                method = method,
                                path = Globals.outpath)

                # PREDICTIONS
                actual_col = f'{var}_actual'
                pred_col = f'{var}_pred'
                train_preds = model.predict(X_train)
                train_pred_true = pd.DataFrame({
                    'pkey' : train_names,
                    actual_col : y_train.flatten().tolist(),
                    pred_col : train_preds.flatten().tolist()
                })
                train_pred_true.to_csv(os.path.join(Globals.outpath, f'{var}_{method}_train_pred_true.csv'), index=False)

                rmsec = sqrt(mean_squared_error(train_pred_true[actual_col], train_pred_true[pred_col]))
                r2_train = model.score(X_train,y_train)
                adj_r2_train = 1 - (1-r2_train)*(len(train_pred_true) - 1) / (len(train_pred_true) - (train_pred_true.shape[1] - 1) - 1)
                 
                # # plot
                plotUtilities.pred_true(df = train_pred_true, 
                            var = var, 
                            method = method, 
                            type = 'train', 
                            rmse = rmsec, 
                            adj_r2 = adj_r2_train, 
                            path = Globals.outpath)

                print(f'\tRMSE-C: {round(rmsec,3)}    R2: {round(r2_train,3)}    Adjusted R2: {round(adj_r2_train,3)}')

                    # optional testing
                if Globals.do_test:
                    print(f'\nTesting model')
                    # TEST PREDICTIONS
                    test_preds = model.predict(X_test)
                    test_pred_true = pd.DataFrame({
                        'pkey' : test_names,
                        actual_col : y_test.flatten().tolist(),
                        pred_col : test_preds.flatten().tolist()
                    })
                    test_pred_true.to_csv(os.path.join(Globals.outpath,f'{var}_{method}_test_pred_true.csv'), index=False)

                    rmsep = sqrt(mean_squared_error(test_pred_true[actual_col], test_pred_true[pred_col]))
                    r2_test = model.score(X_test,y_test)
                    adj_r2_test = 1 - (1-r2_test)*(len(test_pred_true) - 1) / (len(test_pred_true) - (test_pred_true.shape[1] - 1) - 1)
                    
                    # # Plot
                    plotUtilities.pred_true(df = test_pred_true, 
                                var = var, 
                                method = method, 
                                type = 'test', 
                                rmse = rmsep, 
                                adj_r2 = adj_r2_test, 
                                path = Globals.outpath)
                    
                    print(f'\tRMSE-P: {round(rmsep,3)}    R2: {round(r2_test,3)}    Adjusted R2: {round(adj_r2_test,3)}')

                    # write results
                    # take into account analyze.py loq recalculation (or lack of)
                    #TO DO: TEST IF CSV OUTPUT HAS CONSISTENT ROW LENGTH
                    if Globals.do_loq:
                        #no new line, since more will be appended to list
                        outfile.writelines(f'{var},{len(y_train)},{method},{rmsecv},{param},{intercept},{rmsec},{r2_train},{adj_r2_train},{test_fold},{len(y_test)},{rmsep},{r2_test},{adj_r2_test}')
                        if method not in non_linear_methods:
                            recalculation_from_loq_list = Analyze.calculate_loq_median_recalculate_rmse_r2(var,method,Globals.sensitivity, Globals.outpath)
                            recalculation_string = ','.join( str(value) for value in recalculation_from_loq_list)
                            outfile.writelines(f',{recalculation_string}')
                        #if it's a non-linear method, it would be missing 13 cells, so fille them with na
                        else: outfile.writelines(f',NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA')
                        outfile.writelines(f'\n')
                    else: outfile.writelines(f'{var},{len(y_train)},{method},{rmsecv},{param},{intercept},{rmsec},{r2_train},{adj_r2_train},{test_fold},{len(y_test)},{rmsep},{r2_test},{adj_r2_test}\n')
                # write results with no test fold
                # csv output with loq option considered
                else:
                    if Globals.do_loq:
                        outfile.writelines(f'{var},{len(y_train)},{method},{rmsecv},{param},{intercept},{rmsec},{r2_train},{adj_r2_train}') 
                        if method not in non_linear_methods:
                            recalculation_from_loq_list = Analyze.calculate_loq_median_recalculate_rmse_r2(var,method,Globals.sensitivity, Globals.outpath)
                            recalculation_string = ','.join( str(value) for value in recalculation_from_loq_list)
                            outfile.writelines(f',{recalculation_string}')
                        #Method is non-linear, so loq statistics wouldn't print and you'd need 7 NAs
                        else: outfile.writelines(f',NA,NA,NA,NA,NA,NA,NA')
                        outfile.writelines(f'\n')
                    else: outfile.writelines(f'{var},{len(y_train)},{method},{rmsecv},{param},{intercept},{rmsec},{r2_train},{adj_r2_train}\n') 
                #also end elapsed time for each run of variable
                sub_end = time.time()
                print(f'\n{var} took {round(sub_end-sub_start,1)} seconds to run')

        main_end = time.time()
        print(f'\nAll variables took {round((main_end-main_start)/60,1)} minutes to run')
        outfile.close()

    #starts output file(s) with modelling results
    #for each variable...
    # set Globals.data_dict for variable with CV splits
    def apply_models(self):
        # OPEN RESULTS FILE
        outfile = open(os.path.join(Globals.outpath,'modelling_results.csv'), 'w')
        # enter header based on whether or not loq will be called
        if Globals.do_loq:
            if Globals.do_test: #add the extra test statistics from the analyze script to the headers, otherwise just the train stastistics
                outfile.writelines('variable,n_train,model_type,rmsecv,model_params,model_intercept,rmsec,r2_train,adj_r2_train,test_fold,n_test,rmsep,r2_test,adj_r2_test,median_conc_train,loq,n_train_above_loq,median_conc_train_above_loq,rmsec_above_loq,r2_train_above_loq,adj_r2_train_above_loq,median_conc_test,n_test_above_loq,median_conc_test_above_loq,rmsep_above_loq,r2_test_above_loq,adj_r2_test_above_loq\n')
            #loq statistics minus those involving test, since this is without do_test
            else: outfile.writelines('variable,n_train,model_type,rmsecv,model_params,model_intercept,rmsec,r2_train,adj_r2_train,test_fold,n_test,rmsep,r2_test,adj_r2_test,median_conc_train,loq,n_train_above_loq,median_conc_train_above_loq,rmsec_above_loq,r2_train_above_loq,adj_r2_train_above_loq\n')
        else: 
            outfile.writelines('variable,n_train,model_type,rmsecv,model_params,model_intercept,rmsec,r2_train,adj_r2_train,test_fold,n_test,rmsep,r2_test,adj_r2_test\n')
            
        # get elapsed time rather than tqdm
        main_start = time.time()
        for var in Globals.var_to_run:
            # RESET MODEL PARAMETERS
            #maximum number of components for PLS
            max_components_ = Globals.max_components_
            # number of values to test for LASSO, Ridge, ElasticNet, SVR
            num_params_ = Globals.num_params_
            # polynomial degree for SVR and kernel PCR
            poly_deg_ = Globals.poly_deg_
            # maximum number of neighbors for kNN
            max_neighbors_ = Globals.max_neighbors_

            #Get sample test fold
            print(f'\nRunning for {var}')
            fold_col = foldUtilities.get_fold_col(var)
            if Globals.test_fold_per_var[var] is None:
                test_fold = None
            else: test_fold = Globals.test_fold_per_var[var]
            Globals.data_dict, min_samples = foldUtilities.make_data_dict(var, fold_col, test_fold)

            # update parameters if larger than min samples
            print('min samples:', min_samples)
            # max_components and max_neighbors local to run iteration, so Global default doesn't change
            max_components_ = min(len(Globals.spectra), Globals.max_components_) 
            num_params_ = min(Globals.num_params_, min_samples)
            #num_params = min_samples if num_params > min_samples else num_params
            max_neighbors_ = min(min_samples, Globals.max_neighbors_) 

             # functions and arguments per method
            reg_cv_dict = {
                'OLS':{'func':modelUtilities.run_OLS,
                    'args':None},
                'OMP':{'func':modelUtilities.run_OMP,
                    'args':None},
                'LASSO':{'func':parallel.LASSO_parallel, #modelUtilities.run_LASSO,
                        'args':num_params_},
                'Ridge':{'func':parallel.Ridge_parallel,
                        'args':num_params_},
                'ElasticNet':{'func': parallel.ElasticNet_parallel,
                            'args':num_params_},
                'PLS':{'func': modelUtilities.PLS_parallel,       #modelUtilities.run_PLS,
                    'args':max_components_},
                'PCR-lin':{'func':modelUtilities.run_PCR_linear,
                    'args':None},
                'PCR-py':{'func':modelUtilities.run_PCR_poly,
                        'args': poly_deg_},
                'SVR-lin':{'func':modelUtilities.run_SVR_linear,
                        'args':num_params_},
                'SVR-py':{'func':modelUtilities.run_SVR_poly,
                        'args':(num_params_, poly_deg_)},
                'RF':{'func':modelUtilities.run_RF,
                    'args':None},
                'GBR':{'func':modelUtilities.run_GBR,
                    'args':None},
                'kNN':{'func':modelUtilities.run_kNN,
                    'args':max_neighbors_}
            }

            # check length of time for all methods
            #methods_dict wasn't initialized yet
            
            methods_torun = list(Globals.methods_dict[var])
            names = (list(reg_cv_dict.keys()))
            non_linear_methods = ['SVR-py', 'PCR-lin', 'PCR-py', 'RF', 'GBR', 'kNN']
            #axis values for Globals set. You'd have to run prepare script to resample spectra for revised axis
            Globals.axis = list(Globals.spectra['wave'].values)
            #start timer for method
            sub_start = time.time()
            #run selected models for this variable/sample:
            for method in methods_torun:
                # optimize models with CV
                #THIS IS BUILDING/TRAINING MODEL
                print(f'\nPerforming CV for {names[method]}')
                type(reg_cv_dict[names[method]]['args']) 
                #feed in arguments for each method selected
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if type(reg_cv_dict[names[method]]['args'])  == int: # if has arguments...
                        param, rmsecv, model = reg_cv_dict[names[method]]['func'](reg_cv_dict[names[method]]['args'])
                    elif reg_cv_dict[names[method]]['args']:
                        param, rmsecv, model = reg_cv_dict[names[method]]['func'](*reg_cv_dict[names[method]]['args'])
                    else:
                        param, rmsecv, model = reg_cv_dict[names[method]]['func']()
                       # get data in format for full model
                print(f'\nTraining model')
                if Globals.do_test:
                    train_names, X_train, y_train, test_names, X_test, y_test = modelUtilities.format_spectra_meta(var, fold_col, test_fold)
                else:
                    train_names, X_train, y_train =  modelUtilities.format_spectra_meta(var, fold_col)
                #make the method variable the regression method name as opposed to just hte corresponding number of tyhe method
                method = names[method]
                # fit training model, dump asc in outpath destination
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                pickle.dump(model, open(os.path.join(Globals.outpath, f'{var}_{method}_model.asc'), 'wb'), protocol=0)
                # MODEL PARAMETERS
                if method in non_linear_methods:
                    print(f'{method} is non-linear so does not generate coefficients or an intercept')
                    intercept = 'NA'
                else:       
                    # special cases here
                    # the structure for pls may change in teh future
                    # DO TEST TO SEE STRUCTURE ITS IN TO THEN GO WITH THAT
                    if method in ['SVR-lin','PLS']:
                        coef_list = list(model.coef_[0])
                        intercept = model.intercept_[0]               
                    else:
                        coef_list = list(model.coef_)
                        intercept = model.intercept_
                    
                    coef = pd.DataFrame({
                        'wave':Globals.axis,
                        'coef':coef_list
                    })
                    if not pd.api.types.is_numeric_dtype(coef['coef']):
                        coef['coef'] = [x[0] for x in coef.coef]

                    coef.to_csv(os.path.join(Globals.outpath, f'{var}_{method}_coefs.csv'), index=False)

                    # plot
                    plotUtilities.coeffs(df = coef,
                                spectrum = X_train[0],
                                var = var,
                                method = method,
                                path = Globals.outpath)

                # PREDICTIONS
                actual_col = f'{var}_actual'
                pred_col = f'{var}_pred'
                train_preds = model.predict(X_train)
                train_pred_true = pd.DataFrame({
                    'pkey' : train_names,
                    actual_col : y_train.flatten().tolist(),
                    pred_col : train_preds.flatten().tolist()
                })
                train_pred_true.to_csv(os.path.join(Globals.outpath, f'{var}_{method}_train_pred_true.csv'), index=False)

                rmsec = sqrt(mean_squared_error(train_pred_true[actual_col], train_pred_true[pred_col]))
                r2_train = model.score(X_train,y_train)
                adj_r2_train = 1 - (1-r2_train)*(len(train_pred_true) - 1) / (len(train_pred_true) - (train_pred_true.shape[1] - 1) - 1)
                 
                # # plot
                plotUtilities.pred_true(df = train_pred_true, 
                            var = var, 
                            method = method, 
                            type = 'train', 
                            rmse = rmsec, 
                            adj_r2 = adj_r2_train, 
                            path = Globals.outpath)

                print(f'\tRMSE-C: {round(rmsec,3)}    R2: {round(r2_train,3)}    Adjusted R2: {round(adj_r2_train,3)}')

                    # optional testing
                if Globals.do_test:
                    print(f'\nTesting model')
                    # TEST PREDICTIONS
                    test_preds = model.predict(X_test)
                    test_pred_true = pd.DataFrame({
                        'pkey' : test_names,
                        actual_col : y_test.flatten().tolist(),
                        pred_col : test_preds.flatten().tolist()
                    })
                    test_pred_true.to_csv(os.path.join(Globals.outpath,f'{var}_{method}_test_pred_true.csv'), index=False)

                    rmsep = sqrt(mean_squared_error(test_pred_true[actual_col], test_pred_true[pred_col]))
                    r2_test = model.score(X_test,y_test)
                    adj_r2_test = 1 - (1-r2_test)*(len(test_pred_true) - 1) / (len(test_pred_true) - (test_pred_true.shape[1] - 1) - 1)
                    
                    # # Plot
                    plotUtilities.pred_true(df = test_pred_true, 
                                var = var, 
                                method = method, 
                                type = 'test', 
                                rmse = rmsep, 
                                adj_r2 = adj_r2_test, 
                                path = Globals.outpath)
                    
                    print(f'\tRMSE-P: {round(rmsep,3)}    R2: {round(r2_test,3)}    Adjusted R2: {round(adj_r2_test,3)}')

                    # write results
                    # take into account analyze.py loq recalculation (or lack of)
                    #TO DO: TEST IF CSV OUTPUT HAS CONSISTENT ROW LENGTH
                    if Globals.do_loq:
                        #no new line, since more will be appended to list
                        outfile.writelines(f'{var},{len(y_train)},{method},{rmsecv},{param},{intercept},{rmsec},{r2_train},{adj_r2_train},{test_fold},{len(y_test)},{rmsep},{r2_test},{adj_r2_test}')
                        if method not in non_linear_methods:
                            recalculation_from_loq_list = Analyze.calculate_loq_median_recalculate_rmse_r2(var,method,Globals.sensitivity, Globals.outpath)
                            recalculation_string = ','.join( str(value) for value in recalculation_from_loq_list)
                            outfile.writelines(f',{recalculation_string}')
                        #if it's a non-linear method, it would be missing 13 cells, so fille them with na
                        else: outfile.writelines(f',NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA')
                        outfile.writelines(f'\n')
                    else: outfile.writelines(f'{var},{len(y_train)},{method},{rmsecv},{param},{intercept},{rmsec},{r2_train},{adj_r2_train},{test_fold},{len(y_test)},{rmsep},{r2_test},{adj_r2_test}\n')
                # write results with no test fold
                # csv output with loq option considered
                else:
                    if Globals.do_loq:
                        outfile.writelines(f'{var},{len(y_train)},{method},{rmsecv},{param},{intercept},{rmsec},{r2_train},{adj_r2_train}') 
                        if method not in non_linear_methods:
                            recalculation_from_loq_list = Analyze.calculate_loq_median_recalculate_rmse_r2(var,method,Globals.sensitivity, Globals.outpath)
                            recalculation_string = ','.join( str(value) for value in recalculation_from_loq_list)
                            outfile.writelines(f',{recalculation_string}')
                        #Method is non-linear, so loq statistics wouldn't print and you'd need 7 NAs
                        else: outfile.writelines(f',NA,NA,NA,NA,NA,NA,NA')
                        outfile.writelines(f'\n')
                    else: outfile.writelines(f'{var},{len(y_train)},{method},{rmsecv},{param},{intercept},{rmsec},{r2_train},{adj_r2_train}\n') 
                #also end elapsed time for each run of variable
                sub_end = time.time()
                print(f'\n{var} took {round(sub_end-sub_start,1)} seconds to run')

        main_end = time.time()
        print(f'\nAll variables took {round((main_end-main_start)/60,1)} minutes to run')
        outfile.close()

    #use OLS AS TEST METHOD SO THAT SHOULDN'T CHANGE
    #TEST CHOOSE BEST FOLDS FUNCTION TOO
    #


#self an argument for all of the above functions, so object of class supposedly needed
#can always easily change later...
def main(args = sys.argv):
        test = modelSetup()
        test.handle_args()
        test.set_folds()
        testContinued = trainModels()
        if not Globals.parallel:
            testContinued.apply_models_non_parallel()
        else:
            testContinued.apply_models()
        if Globals.do_loq: os.rename(os.path.join(Globals.outpath,'modelling_results.csv'), os.path.join(Globals.outpath,'modelling_results_with_loq.csv') )

if __name__ == '__main__':
    main()
