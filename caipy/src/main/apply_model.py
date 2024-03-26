import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import re
import argparse
# hide warning about invalid escape sequence in prompts
import warnings
warnings.simplefilter('ignore', SyntaxWarning)
from utilities import dataLoadingUtilities, generalUtilities, plotUtilities

#from model_tools import check_csv, check_asc, make_bool, convert_spectra, Plot


'''
by Cai Ytsma (cai@caiconsulting.co.uk)
Last updated 12 November 2023

Apply .asc sklearn model to input data. Returns .csv of predicted values.
Optionally include metadata file for test samples to generate:
    - RMSE-P, R2, adjusted R2
    - predicted vs. true scatter plot

Spectra file column format:
'wave' (wavelength axis), {sample_1} (spectral intensities), {sample_2}, etc.

Metadata file format:
'pkey' (sample names), {variable} (values to be predicted), optionally 'Sample Name' or 'Sample_Name'
'''
#-----------------#
# HELPER FUNCTION #
#-----------------#


# TO DO: https://github.com/ytsmacr/caipy/commit/fe3a4000009963d9046a978eedbddd6a278310de INTEGRATE (IF NO META, MAKE ONE WITH JUST SPECTRA NAMES AND PREDICTIONS, SO HAVE COMPS PROVIDED)
def check_model_file(file):
    while not os.path.exists(file):
        print(f'Error: path {file} does not exist')
        file = dataLoadingUtilities.check_asc(input(file))
    return file
def main(sysargs = None): 
    #-------------------------------------------------------------#
    #                     DEFINED VARIABLES                       #
    #-------------------------------------------------------------#
    # PROMPTS
    single_prompt = r"Are you applying more than one model? (y/n): "
    model_prompt = r"Model file path: (e.g. C:\Documents\SiO2_model.asc) "
    model_list_prompt = r"Model file paths, separated by a comma: (e.g. C:\Documents\SiO2_model.asc,C:\Documents\Al2O3_model.asc,C:\Documents\Sr_model.asc) "
    spectra_prompt = r"Test spectra file path: (e.g. C:\Documents\spectra.csv) "
    comps_prompt = r"Do you have compositions for these samples (y/n): "
    has_comps_prompt = r"Test metadata file path: (e.g. C:\Documents\metadata.csv) "
    out_prompt = r"File path to export results: "

    #-------------------#
    # INPUT INFORMATION #
    #-------------------#

    # from arguments
    #TO DO: IF YOU STATE NO COMPS, SHOULD NOT PROMPT META DATA  
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', type=str, default=None, help=model_prompt) # for a single model
    parser.add_argument('-ml', '--model_list', type=str, default=None, help=model_list_prompt) # for multiple models
    parser.add_argument('-sf', '--spectra_file', type=str, default=None, help=spectra_prompt)
    parser.add_argument('-hc', '--have_comps', type=bool, default=None, help='Boolean of whether the test data have compositions')  #reword as comps
    parser.add_argument('-mf', '--meta_file', type=str, default=None, help=has_comps_prompt)
    parser.add_argument('-o', '--outpath', type=str, default=None, help=out_prompt)

    args, unknown = parser.parse_known_args(sysargs)

    #TO DO: REMOVE
    have_comps  = False
    model_file = args.model_file
    if model_file is not None:
        model_file = model_file.replace("'","")
    model_list = args.model_list
    if model_list is not None:
        model_list = [str(item) for item in args.model_list.split(',')]
        print(model_list)
    spectra_file = args.spectra_file
    if spectra_file is not None:
        spectra_file = spectra_file.replace("'","")
    outpath = args.outpath
    if outpath is not None:
        outpath = outpath.replace("'","")
    meta_file = args.meta_file
    if meta_file is not None:
        meta_file = meta_file.replace("'","")
        # if have meta, automatically has comps
        have_comps = True
    if have_comps is not None:
        have_comps = args.have_comps

    # GET FILE INFORMATION
    # model
    if (model_file is None) and (model_list is None):
        # see if single or multiple
        single_model =generalUtilities.make_bool(input(single_prompt).lower())
        while single_model == 'error':
            print('Error: Input needs to be either y or n')
            single_model = generalUtilities.make_bool(input(single_prompt).lower())
        if single_model is True:
            model_file = dataLoadingUtilities.check_asc(input(model_prompt))
            model_file = check_model_file(model_file) # check it exists
            model_list = [model_file] # convert to list to streamline procedure
        elif single_model is False:
            model_list = input(model_list_prompt)
            # prep files
            model_list = model_list.split(',')
            model_list = [check_model_file(x) for x in model_list] # check they all exist

    if model_list is None:
        model_list = [model_file]

    print('***REMINDER***\nTest data should be processed identically to how training data were processed\n')

    # spectra
    if spectra_file is None:
        spectra_file = dataLoadingUtilities.check_csv(input(spectra_prompt))
    while not os.path.exists(spectra_file):
        print(f'Error: path {spectra_file} does not exist')
        spectra_file = dataLoadingUtilities.check_csv(input(spectra_prompt))
    spectra = pd.read_csv(spectra_file)

    # have compositions for test samples?
    if (have_comps is None) and (meta_file is None):
        have_comps = generalUtilities.make_bool(input(comps_prompt).lower())
        while have_comps == 'error':
            print('Error: Input needs to be either y or n')
            have_comps = generalUtilities.make_bool(input(comps_prompt).lower())
    # if so, get comps
    if have_comps is True:
        #CHANGE AROUND HERE SO THAT IF NO META FILE, MAKE ONE WITH JUST SPECTRA NAMES AND PREDICTIONS (SO HAVE COMPS PROVIDED)
        
        if meta_file is None:
            meta_file = dataLoadingUtilities.check_csv(input(has_comps_prompt))
            while not os.path.exists(meta_file):
                print(f'Error: path {meta_file} does not exist')
                meta_file = dataLoadingUtilities.check_csv(input(comps_prompt))
            
    # folder to export results to
    if outpath is None:
        outpath = input(out_prompt)
    # make it if it doesn't exist
    if not os.path.exists(outpath):
        os.mkdir(outpath)
        
    #get list of models and variables as dictionry:
    for file in model_list:
        file = os.path.basename(os.path.normpath(file))
        print("file in model list", file)
        file_params = re.split(r'[_ .]',file)
        print(file_params)
        #file in model list TiO2_PLS_model.asc ['TiO2', 'PLS', 'model', 'asc']

    # load files
    spectra = pd.read_csv(spectra_file)
    if have_comps:
        meta = pd.read_csv(meta_file)
        pred_df = meta[['pkey']].copy()
    else:
        samples = spectra.columns.values
        samples = np.delete(samples, (0), axis = 0)
        pred_df = pd.DataFrame(  {'pkey': samples})
       # print(pred_df)

    #TO DO: RENAMEEEEEE OUTPUTS
    if have_comps is True:
        # OPEN RESULTS FILE
        outfile = open(os.path.join(outpath,'prediction_results.csv'), 'w') #metadata_with_prediction_results.csv
        # enter header
        outfile.writelines('variable,model_type,n_pred,rmsep,r2_test,adj_r2_test\n')
    else:
        # OPEN RESULTS FILE
        outfile = open(os.path.join(outpath,'prediction_results_no_comps.csv'), 'w') #prediction_results.csv
        # enter header (maybe come up with better ones...)
        #TO DO:
    #    columns = ['pkey', f'Predicted {variable} by {model_type} model']
        outfile.writelines('pkey,model_type,n_pred\n')
    # RUN
    for i, model_file in enumerate(model_list):
        model = pickle.load(open(model_file, 'rb'))
        # format data
        X_test = dataLoadingUtilities.convert_spectra(spectra)
        # get predictions
        test_pred = model.predict(X_test)
        
        # get variable and model information from filename if possible
        model_name = os.path.split(model_file)[1]
        if re.match(r".+_.+_model\.asc", model_name):
            var = model_name.split('_')[0]
        else:
            var = input('Error: Could not extract variable name from model filename.\nWhat is the relevant variable? ')
        
        if re.match('.+_.+_model.asc$', model_name):
            method = model_name.split('_')[1]
        else:
            method = input('Error: Could not extract regression method from model filename.\nWhat is the relevant method? ')

        # prep column names
        actual_col = f'{var}_actual'
        pred_col = f'{var}_pred'
        
        if have_comps:
            # append results to df #QUESTION: SHOULD PRED_DF ONLY BE DEFINED AND TOUCHED IF COMPS PROVIDED? WHAT WAS THIS REALLY EVEN PREVIOUSLY USED FOR?
            pred_df[pred_col] = list(test_pred.flatten())
            # see if var in meta file
            count = 0
            while var not in meta.columns:
                if count >= 1:
                    var_cols = ', '.join([col for col in meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)])
                    print('Metadata columns to choose from: ', var_cols)
                var = input(f'Error: {var} not a metadata column. What is the relevant variable? ')
                count += 1
        
            # add actual values
            for option in ['Sample_Name', 'Sample Name']:
                if option in meta.columns:
                    if i == 0:
                        cols_to_add = ['pkey', option, var]
                    else:
                        cols_to_add = ['pkey', var]
            pred_df = pred_df.merge(meta[cols_to_add], how='left', on='pkey')
            pred_df.rename(columns={var:actual_col}, inplace=True)

            # calculate metrics on temp df
            pred_true = pred_df[[actual_col,pred_col]].copy().dropna()
            if len(pred_true) > 2:
                # RMSE-P
                rmsep = sqrt(mean_squared_error(pred_true[actual_col],
                                                pred_true[pred_col]))
                # R2
                r2 = r2_score(pred_true[actual_col],
                            pred_true[pred_col])
                # adjusted r2
                adj_r2 = 1 - (1-r2)*(len(pred_true) - 1) / (len(pred_true) - (pred_true.shape[1] - 1) - 1)
                
                # add results
                outfile.writelines(f'{var},{method},{len(pred_true)},{rmsep},{r2},{adj_r2}\n')
                
                # PLOT
                plotUtilities.pred_true(df = pred_true,
                            var = var, 
                            method = method, 
                            type = 'test',
                            rmse = rmsep,
                            adj_r2 = adj_r2,
                            path = outpath)
            else:
                print(f'{var} only has {len(pred_true)} sample(s) to test, so skipping RMSE and R2 calculations')
                # add results
                outfile.writelines(f'{var},{method},{len(pred_true)},,,\n')
        else:
            pred_df[pred_col] = list(test_pred.flatten())
            outfile.writelines(f'{var},{method},{len(test_pred.flatten())}\n')


    if have_comps:
        outfile.close()
        df = pd.DataFrame(pred_df)
        df.to_csv(os.path.join(outpath, f'predictions_with_comps.csv'), index=False)
        print('Exported predicted vs. true values and plot')
    else:
        df = pd.DataFrame(pred_df)
        df.to_csv(os.path.join(outpath, f'predictions.csv'), index=False)
        print('Exported predicted values')


if __name__ == '__main__':
    main()