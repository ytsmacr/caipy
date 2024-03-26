# standard packages
import argparse
import sys
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from statistics import mean, median
from tqdm import tqdm
import os
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from globals import *
from utilities import dataLoadingUtilities, generalUtilities
plt.set_loglevel('error')



class Analyze():
    #post modeling, calculates sensitivies, adds columns to result file with new rmse after loq calculation
    #integrate into pipeline, but needs sensitivies (would rely on actual sensitivity values, how do we supply sensitivites)
    #sensitivity can just be a value someone supplies:
    #calculate loq after buildingb models?
    #here we provide values to do so
    # post modeling analysis function
    # generate sensitivity funciotn

    
#check https://github.com/ytsmacr/LIBS-modelling/blob/main/calculate_sensitivities.ipynb 

    '''
    Standalone functions for commonly used analyses
    '''
    # There is context to consider rmse: how reproducable lab data is
    # sensitiviy = reproducablility of instrument. LIBS isn't very reproducable, so there is uncertainty form the noise and fluctuations of data
    # determine sensitivity by looking at how different spectra of same sample are, std deviation and get average for each spectra
    # How we distinguish signal from noise, is there a signal or not 
    # for multivariate, multiply senstivity value by model corefficients 
    # requires us to use linear models
    # limit of blank, limit of detection, limit of quantification for signal certainty
    # we calculate at what level you can start to quantify something with a model,
    #     if data is noisy, loq takes constant sensitivity and converts it into a vector 
    #     by multiplying lob factor 1, lod factor 3, and other factors to be sure we're 
    #     above that noise and make sure we're mreasuing things corretly
    # Then we go back to predictions, anything below the loq is discarded because of hte uncertaintly. (loq is like the bare minimum to make a detection significant)
    # We then recaulculate rmse from these values
    # rmse will be higher after removing what's below loq, you're kind of biasing up
    # So we use median rather than mean so that it can be contextualized for comparison
    # Ex: elements with very low abundances in rocks, but hard to make calibration when there's so little
    # So rocks were doped to set concentrations, but we don't want these few high samples to skew mean, so median used
    


    def calculate_loq_median_recalculate_rmse_r2(
        variable:str, # variable ID'd in the model output files
        reg_method: str, # regression method ID'd in the model output files
        sensitivity: float, # instrument sensitivity
        folder_path: str # path of folder containing model output files
        ):
    
        '''
        This function comes after spectral_regression_modelling and calculates LOQ
        along with a few other things. The resulting dictionary can be simply merged
        with the standard results file after converting to DataFrame. 
        
        I'm inclined to either integrate this with the spectral_regression_modelling
        procedure, or keep it distinct because the user also needs to feed in the 
        sensitivity. For now, I think I'm happy to keep them separate - maybe this 
        function will become obsolete if/when I integrate it.

        Will have to think about how the instrument sensitivities are stored or
        fed if becomes integrated.
        '''

        print('**NOTE**: Sensitivity must be calculated from spectra processed identically to model training spectra')

        # every RMSE should change with the LOQ, but don't know how I could calculate it for the RMSECV

        # prep files
        # TO DO: test has_test properly recieved
        coeff_df = pd.read_csv(os.path.join(folder_path, f'{variable}_{reg_method}_coefs.csv'))
        train_df = pd.read_csv(os.path.join(folder_path, f'{variable}_{reg_method}_train_pred_true.csv'))
        try:
            test_df = pd.read_csv(os.path.join(folder_path, f'{variable}_{reg_method}_test_pred_true.csv'))
            has_test = True
        except:
            has_test = False
        
        # calculate LOQ
        vector = pow(pow(coeff_df['coef'], 2).sum(),0.5)  #square root of sum of squares
        loq = 10 * vector * sensitivity

        pred = f'{variable}_pred'
        true = f'{variable}_actual'

        #median before
        print(f"loq value for {variable} with {reg_method}:", loq)
        train_median = round(median(train_df[true].values),2)
        # remove those below LOQ
        train_above_loq = train_df[train_df[pred]>loq].copy()
        #TO DO: this this that throw an error if train above loq has size 0, which can happen for certain sensitivities? like sensitivity =2 for test_meta_stratified and test_spectra
        if train_above_loq.empty:
            message = f"Calculated loq of {loq} is considered high since there isn't any data in the file '{variable}_{reg_method}_train_pred_true.csv' that has predictuions above the loq value. Can't calculate revised rmsec"
            raise ValueError(message)
        rev_rmsec = sqrt(mean_squared_error(train_above_loq[true], train_above_loq[pred]))
        rev_train_r2 = r2_score(train_above_loq[true], train_above_loq[pred])
        rev_train_adj_r2 = 1 - (1-rev_train_r2)*(len(train_above_loq) - 1) / (len(train_above_loq) - (train_above_loq.shape[1] - 1) - 1)
        rev_train_median = round(median(train_above_loq[true].values),2)
        rev_n_train = len(train_above_loq)

        if has_test:
            #median before
            test_median = round(median(test_df[true].values),2)
            # remove those below LOQ
            test_above_loq = test_df[test_df[pred]>loq].copy()
            # recalculate metrics
            rev_rmsep = sqrt(mean_squared_error(test_above_loq[true], test_above_loq[pred]))
            rev_test_r2 = r2_score(test_above_loq[true], test_above_loq[pred])
            rev_test_adj_r2 = 1 - (1-rev_test_r2)*(len(test_above_loq) - 1) / (len(test_above_loq) - (test_above_loq.shape[1] - 1) - 1)
            rev_test_median = round(median(test_above_loq[true].values),2)
            rev_n_test = len(test_above_loq)

            # cumulate results
            # results were written so that in teh case that do_test is true but there are one or more samples that
            # only have 1 fold (meaning they wouldn't actually have a test fold pulled out), their results dict can be
            # written to the modeling results cv with all the columns with test statistics left blank
            # Therefore, it is crucial to keep thingsi n this order, with train statistics and then test statistics
            
            #result_dict kept for now, but result_list oficially used for the sake of adding to csv output in mnodels.py
            result_dict = {
                'median_conc_train':train_median,
                'loq':loq,
                'n_train_above_loq':rev_n_train,
                'median_conc_train_above_loq':rev_train_median,
                'rmsec_above_loq':rev_rmsec,
                'r2_train_above_loq':rev_train_r2,
                'adj_r2_train_above_loq':rev_train_adj_r2,
                'median_conc_test':test_median,
                'n_test_above_loq':rev_n_test,
                'median_conc_test_above_loq':rev_test_median,
                'rmsep_above_loq':rev_rmsep,
                'r2_test_above_loq':rev_test_r2,
                'adj_r2_test_above_loq':rev_test_adj_r2
            }
            result_list = [
                train_median,
                loq,
                rev_n_train,
                rev_train_median,
                rev_rmsec,
                rev_train_r2,
                rev_train_adj_r2,
                test_median,
                rev_n_test,
                rev_test_median,
                rev_rmsep,
                rev_test_r2,
                rev_test_adj_r2
            ]
        else:
            # cumulate results
            result_dict = {
                'median_conc_train':train_median,
                'loq':loq,
                'n_train_above_loq':rev_n_train,
                'median_conc_train_above_loq':rev_train_median,
                'rmsec_above_loq':rev_rmsec,
                'r2_train_above_loq':rev_train_r2,
                'adj_r2_train_above_loq':rev_train_adj_r2
            }
            result_list = [
                train_median,
                loq,
                rev_n_train,
                rev_train_median,
                rev_rmsec,
                rev_train_r2,
                rev_train_adj_r2,
            ]
        
        # finally, export the list so that it can be added to csv
        return result_list


def main(sysargs = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outpath', type=str, default=None, help='Path of folder with modelling output results ')
    parser.add_argument('-sen', '--sensitivity', type=float, default=None, help='Float value for the sensitivty value of the calibration instrument,')
    parser.add_argument('-dt', '--do_test', type=bool, default=None, help='Hold a fold out as test data (bool)')
    args, unknown = parser.parse_known_args(sysargs)
    Globals.outpath = args.outpath
    if Globals.outpath is None:
        outpath, all_files = dataLoadingUtilities.get_out_folder()
        Globals.outpath = outpath
    Globals.sensitivity = args.sensitivity
    if Globals.sensitivity is None:
        Globals.sensitivity = float(input("Provide a float value for the sensitivty value of the calibration instrument, for loq calculations: "))
    Globals.do_test = args.do_test
    if Globals.do_test is None:
            Globals.do_test = generalUtilities.make_bool(input('Was a fold held out exclusively for test data? y for yes, n for no: '))
    non_linear_methods = ['SVR-py', 'PCR-lin', 'PCR-py', 'RF', 'GBR', 'kNN']
    #resulting outfile opened: 
    outfile = open(os.path.join(Globals.outpath,'modelling_results_with_loq.csv'), 'w')
    #write headers...
    if Globals.do_test: #add the extra test statistics from the analyze script to the headers, otherwise just the train stastistics
            outfile.writelines('variable,n_train,model_type,rmsecv,model_params,model_intercept,rmsec,r2_train,adj_r2_train,test_fold,n_test,rmsep,r2_test,adj_r2_test,median_conc_train,loq,n_train_above_loq,median_conc_train_above_loq,rmsec_above_loq,r2_train_above_loq,adj_r2_train_above_loq,median_conc_test,n_test_above_loq,median_conc_test_above_loq,rmsep_above_loq,r2_test_above_loq,adj_r2_test_above_loq\n')
    #loq statistics minus those involving test, since this is without do_test
    else: outfile.writelines('variable,n_train,model_type,rmsecv,model_params,model_intercept,rmsec,r2_train,adj_r2_train,test_fold,n_test,rmsep,r2_test,adj_r2_test,median_conc_train,loq,n_train_above_loq,median_conc_train_above_loq,rmsec_above_loq,r2_train_above_loq,adj_r2_train_above_loq\n')
    #read in modelling_results.csv
    df= pd.read_csv(os.path.join(Globals.outpath, f'modelling_results.csv'))
    for index, row in df.iterrows():
        if row['model_type'] in non_linear_methods:
            print("Unable to run analysis function for", row['variable'], "due to non-linear nature of method", row['model_type'])
            line = ','.join( str(value) for value in row)
            outfile.write(f'{line}\n')
        else:
            #if valid type of method, run analysis script and write row contents + values into outfile
            line = ','.join( str(value) for value in row)
            outfile.write(f'{line}')
            recalculation_from_loq_list = Analyze.calculate_loq_median_recalculate_rmse_r2(row['variable'],row['model_type'],Globals.sensitivity, Globals.outpath)
            recalculation_string = ','.join( str(value) for value in recalculation_from_loq_list)
            outfile.writelines(f',{recalculation_string}\n')
    outfile.close()

if __name__ == '__main__':
    main()