#general packages
import argparse
import statistics
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from statistics import mean, median
from tqdm import tqdm
import os
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
plt.set_loglevel('error')

# preprocessing
from tools.airPLS import airPLS
from tools.spectres import spectres
from sklearn.preprocessing import normalize

#parallization tools:
from multiprocessing import Pool
from joblib import Parallel, delayed

# modelling
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, OrthogonalMatchingPursuit
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from globals import Globals
import stratify_samples 


#class preprocessingUtilities for spectra
#class cleaningUtiltiies for meta

class preprocessingUtilities:
    #after spectra preprocessing options from spectra_options() received, go through the options and applyu them to Globals.spectra
    def apply_spectra_args(spectra):
        if Globals.resample is not None:
            if Globals.resample != 1 and Globals.resample != 2:
                raise ValueError
            #resample to match
            if int(Globals.resample) == 1:
                full_path_of_sample = input("Enter full path of a spectra sample to match spectra path to. Hit ENTER to default resampling to SuperCam -2px shifted wavelength axis: ")
                if not full_path_of_sample:
                    Globals.spectra = preprocessingUtilities.resample_to_match(spectra)
                else:
                    sample_to_match = pd.read_csv(full_path_of_sample, index_col=False)
                    Globals.spectra = preprocessingUtilities.resample_to_match(spectra, sample_to_match)
                print("spectra resampled")
            #resample to min step
            else:
                preprocessingUtilities.resample_to_min_step(Globals.spectra)
        #BLR
        if Globals.baseline == True:
            Globals.spectra = preprocessingUtilities.AirPLS(spectra)
            print("Applied airPLS baseline removal to spectra dataframe")
        #Normalization
        if Globals.normalization is not None:
            match int(Globals.normalization):
                case 0:
                    print("SuperCam method normalization...")
                    Globals.spectra = preprocessingUtilities.norm5_SC(Globals.spectra)
                case 1:
                    print("ChemCam method normalization...")
                    Globals.spectra = preprocessingUtilities.norm3_CL(Globals.spectra)

                case 2:
                    print("SuperLIBS 10K method normalization...")
                    Globals.spectra = preprocessingUtilities.norm3_SL_10K(Globals.spectra)
                case 3:
                    print("SuperLIBS 18K method normalization...")
                    Globals.spectra = preprocessingUtilities.norm3_SL_18K(Globals.spectra)
                case _:
                    raise ValueError
        Globals.spectra.to_csv(Globals.spectra_path, index = False)
    
    #processes spectra inputs
    def spectra_options(spectra):
        if spectra is None:
            raise AttributeError("No spectra data file")
        blr_prompt = 'Use airPLS baseline removal on spectra data? y for yes, n for no: '
        if Globals.baseline is None:
            ans = input(blr_prompt)
            Globals.baseline = generalUtilities.make_bool(ans)
        resample_prompt = '''Resample data to match [1] or to min step [2]
        Hit ENTER for no resampling of data: '''
        if Globals.resample is None:
            ans = input(resample_prompt)
            if not ans:
                Globals.resample = None
            else: 
                Globals.resample = ans
        normalization_prompt = '''Normalize spectra data using method selection from:
            [0] SuperCam method
            [1] ChemCam method
            [2] SuperLIBS 10K method
            [3] SuperLIBS 18K method
            hit ENTER for none of the options (no normalization): '''
        if Globals.normalization is None:
            ans = input(normalization_prompt)
            if not ans:
                Globals.normalization= None
            else: 
                Globals.normalization = ans

    '''
    Standalone functions used to preprocess spectra

       
    RESAMPLE SPECTRA
    '''

    # resample spectra to given axis
    def resample_to_match(spectra, spectra_to_match = None):

        spectra_to_resample = spectra.copy()
        
        # using Spectres package
        ## https://spectres.readthedocs.io/en/latest/
        ## https://arxiv.org/pdf/1705.05165v1.pdf

        if spectra_to_match is None:
            print('Resampling to SuperCam -2px shifted wavelength axis by default')
            spectra_to_match = pd.read_csv(os.path.join('data','SuperCam_cal_shift-2pix_axis.csv'), index_col=False)

        # if input is a dataframe
        if isinstance(spectra_to_match,pd.DataFrame):
            if 'wave' not in spectra_to_match.columns:
                print('Axis to match needs "wave" header')
                return
            new_axis = spectra_to_match['wave'].to_numpy()

        # if input is an array
        elif isinstance(spectra_to_match,np.ndarray):
            new_axis = spectra_to_match

        else:
            print('Spectra to match must either be a dataframe with "wave" column, or a numpy array of values')
            return

        old_axis = spectra_to_resample['wave'].to_numpy()

        if list(new_axis) == list(old_axis):
            print('Spectral axes already matched')
            return

        spectra_to_resample.drop('wave', axis=1, inplace=True)
        old_spectra = spectra_to_resample.T.to_numpy()

        new_spectra = spectres(new_axis, old_axis, old_spectra, fill=0, verbose=False)
        new_spectra = pd.DataFrame(new_spectra).T
        new_spectra.columns = spectra_to_resample.columns
        new_spectra.insert(0,'wave',new_axis)
        return new_spectra
    #what is wave?
    # resample uniformly to minimum step size
    def resample_to_min_step(spectra):
        
        spectra_to_resample = spectra.copy()

        if 'wave' not in spectra_to_resample.columns:
            print('Input spectra must have "wave" axis column')
            return

        axis = spectra_to_resample['wave'].to_numpy()
        # REPLACE WAVE WITH AXIS
        # get step sizes
        step_set = set()
        for i in np.arange(len(axis))[:-1]:
            current = axis[i]
            next = axis[i+1]
            step = next-current
            step_set.add(step)

        # get minimum
        min_step = min(step_set)

        # populate new axis with this step size
        min_step_axis = np.arange(start = axis[0], stop = axis[-1]+min_step, step=min_step)

        # resample spectra to match this
        resampled_spectra = preprocessingUtilities.resample_to_match(spectra_to_resample, min_step_axis)
        return resampled_spectra

    '''
    BASELINE REMOVAL
    '''

    # airPLS baseline removal
    ## recommend that resample to min step size first ##
    def AirPLS(spectra,
               l = 100):

        spectra_to_blr = spectra.copy()
        
        if spectra_to_blr.isnull().values.any():
            raise ValueError('The spectra contains NA values - remove and rerun')
        
        if spectra_to_blr.columns[0] != 'wave':
            raise ValueError('This function needs the first column to be the axis, "wave"')

        spec_list = []

        for column in spectra_to_blr.columns[1:]:
            spectrum = spectra_to_blr[column]
            bl = airPLS(spectrum, lambda_ = 1)
            blr_spectrum = spectrum - bl
            blr_spectrum = blr_spectrum.tolist()
            spec_list.append(blr_spectrum)

        blr_spectra = pd.DataFrame(spec_list).T
        blr_spectra.columns = spectra_to_blr.columns[1:]
        blr_spectra.insert(0, 'wave', spectra_to_blr['wave'])

        return blr_spectra
        
    '''
    NORMALIZATION
    '''

    # normalize each df subset of data, then concatenate
    def normalize_regions(df_list: list,
                          method = 'l1'):

        count = 0

        # default, but leaving option open for other methods
        if method == 'l1':
            def normalization(array):
                return (array/sum(array))
        else:
            print('Method not defined')
            return

        for df in df_list:
            spectra_list = []

            # first, add wavelength
            wave = list(df['wave'])
            spectra_list.append(wave)

            # get names
            names = df.columns

            for sample in df.columns[1:]:
                # convert spectrum to array
                spectrum = df[sample].T.to_numpy()
                # normalize spectrum
                norm_spectrum = normalization(spectrum)
                # add to list
                spectra_list.append(norm_spectrum)

            # then, make df or add to df
            if count == 0:
                normed_dataset = pd.DataFrame(spectra_list).T
                normed_dataset.columns = names

            else:
                df_to_add = pd.DataFrame(spectra_list).T
                df_to_add.columns = names

                normed_dataset = pd.concat([normed_dataset, df_to_add], ignore_index=True)

            count+=1

        return normed_dataset

    # normalize by SuperCam method
    def norm5_SC(spectra):
        spectra_tonorm = spectra.copy()

        # limits from Anderson et al. 2021, Table 1.
        # https://doi.org/10.1016/j.sab.2021.106347

        uv = spectra_tonorm[(spectra_tonorm['wave'] >= 243.79) & (spectra_tonorm['wave'] <= 341.36)].copy()
        vis = spectra_tonorm[(spectra_tonorm['wave'] >= 379.26) & (spectra_tonorm['wave'] <= 464.54)].copy()
        vnir_1 = spectra_tonorm[(spectra_tonorm['wave'] >= 537.57) & (spectra_tonorm['wave'] <= 619.82)].copy()
        vnir_2 = spectra_tonorm[(spectra_tonorm['wave'] >= 620.08) & (spectra_tonorm['wave'] <= 712.14)].copy()
        vnir_3 = spectra_tonorm[(spectra_tonorm['wave'] >= 712.17) & (spectra_tonorm['wave'] <= 852.77)].copy()

        df_list = [uv, vis, vnir_1, vnir_2, vnir_3]

        normed_spectra = preprocessingUtilities.normalize_regions(df_list)

        return normed_spectra

    # normalize by ChemCam method
    def norm3_CL(spectra):
        
        spectra_tonorm = spectra.copy()

        uv = spectra_tonorm[(spectra_tonorm['wave'] >= 246.68) & (spectra_tonorm['wave'] <= 338.42)].copy()
        vis = spectra_tonorm[(spectra_tonorm['wave'] >= 387.9) & (spectra_tonorm['wave'] <= 469.1)].copy()
        vnir = spectra_tonorm[(spectra_tonorm['wave'] >= 492.65) & (spectra_tonorm['wave'] <= 849.1)].copy()

        df_list = [uv, vis, vnir]

        normed_spectra = preprocessingUtilities.normalize_regions(df_list)

        return normed_spectra

    # normalize by SuperLIBS 10K method
    def norm3_SL_10K(spectra):
        
        spectra_tonorm = spectra.copy()

        uv = spectra_tonorm[(spectra_tonorm['wave'] >= 233.12) & (spectra_tonorm['wave'] <= 351.35)].copy()
        vis = spectra_tonorm[(spectra_tonorm['wave'] >= 370.16) & (spectra_tonorm['wave'] <= 479.07)].copy()
        vnir = spectra_tonorm[(spectra_tonorm['wave'] >= 498.14) & (spectra_tonorm['wave'] <= 859.44)].copy()

        df_list = [uv, vis, vnir]

        normed_spectra = preprocessingUtilities.normalize_regions(df_list)

        return normed_spectra

    # normalize by SuperLIBS 18K method
    def norm3_SL_18K(spectra):
        
        spectra_tonorm = spectra.copy()

        uv = spectra_tonorm[(spectra_tonorm['wave'] >= 233.12) & (spectra_tonorm['wave'] <= 351.35)].copy()
        vis = spectra_tonorm[(spectra_tonorm['wave'] >= 370.16) & (spectra_tonorm['wave'] <= 479.07)].copy()
        vnir = spectra_tonorm[(spectra_tonorm['wave'] >= 508.3) & (spectra_tonorm['wave'] <= 869.2)].copy()

        df_list = [uv, vis, vnir]

        normed_spectra = preprocessingUtilities.normalize_regions(df_list)

        return normed_spectra

    
class dataLoadingUtilities:
    # convert spectra df to np array for modelling
    def convert_spectra(spectra):
        first_col = spectra.columns[0]
        if first_col != 'wave':
            cont = generalUtilities.make_bool(input(f'Warning: convert_spectra assumes the first column is the wavelength axis and ignores it. The first column of your data is {first_col}. Continue? (y/n):'))
            if not cont:
                raise ValueError('Aborting')
        conv_spectra = np.array(spectra[spectra.columns[1:]].T)
        return conv_spectra
    def check_asc(filename):
        if filename[-4:] != '.asc':
            filename = filename + '.asc'

        return filename
    #processes the Globals.methods_dict
    def handle_standard():
         # PROMPTS
        std_prompt = 'Should all variables follow the same modelling procedure? (set test fold, model type(s)) (y/n): '
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
        #CORRESPONDING OPTION DICTIONARY:
        #Doesn't fully corresond with reg_cv_dict (which does't have an all methods entry, so take 0 for all numbers from 1 to 12,
        #and 1-13 as each number pegged down by 1)
        method_options = {
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
        # check if run same procedure for all var
        if Globals.standard is None:
            Globals.standard = generalUtilities.make_bool(input(std_prompt).lower())
            while Globals.standard == 'error':
                print('Error: Input needs to be either y or n')
                Globals.standard = generalUtilities.make_bool(input(std_prompt).lower())
        #methods_dict for standard
        if Globals.standard == True:
            print("Selected standard procedure for sample variable")
            method_type = input(method_prompt)
            div = ',' if ',' in method_type else ' '
            if "0" in method_type and "10" not in method_type and len(method_type) > 1:
                raise KeyError("Invalid selection. If you want to run all methods, 0 is the only input needed")
            #decrement selection by 1
            if method_type  == "0":
                method_type = list(range(13))
            else: method_type = [(int(x) -1) for x in method_type.split(div)]
            #method options should jsut be the numbers 1 - 12
            if set(method_type).issubset(set(method_options.keys())):
                methods_torun = list(set(method_type))
                print("\nPerforming regression number(s)", ', '.join([str(x + 1) for x in methods_torun]))
            else:    
                raise KeyError(f"Error: Input must be one or more separeted selections of {method_options.keys()}")
            for var in Globals.var_to_run:
                Globals.methods_dict[var] = set(method_type)
        #methods_dict for non-standard
        else:
            for var in Globals.var_to_run:
                overall_prompt = "For sample variable " +  var + " " + method_prompt
                method_type = input(overall_prompt)
                div = ',' if ',' in method_type else ' '
                if "0" in method_type and len(method_type) > 1:
                    raise KeyError("Invalid selection. If you want to run all methods, 0 is the only input needed")
                #decrement selection by 1
                if method_type  == "0":
                    method_type = list(range(13))
                else: method_type = [(int(x) -1) for x in method_type.split(div)]
                #method options should jsut be the numbers 1 - 12
                if set(method_type).issubset(set(method_options.keys())):
                    methods_torun = list(set(method_type))
                    print("\nPerforming regression number(s)", ', '.join([str(x + 1) for x in methods_torun]))
                else:    
                    raise KeyError(f"Error: Input must be one or more separeted selections of {method_options.keys()}")
                Globals.methods_dict[var] = set(method_type)
    #processes do_test and do_loq
    def models_args():
        #check if do_test, which is none by default
        if Globals.do_test is None:
            Globals.do_test = generalUtilities.make_bool(input('Hold a fold out as exclusively test data? y for yes, n for no: '))
            while Globals.do_test == 'error':
                print('Error: Input needs to be either y or n')
                Globals.do_test=  generalUtilities.make_bool(input('Hold a fold out as exclusively test data? y for yes, n for no: '))
        #asks if Analyze.calculate_loq_median_recalculate_rmse_r2() shoudl be run, and then it prompts for sensitivity value if yes
        if Globals.do_loq is None:
            Globals.do_loq = generalUtilities.make_bool(input('Post modelling function: Calculate LOQ of each sample for each regression model, and then recalculate rmse and r2 statistics from the train and test prediction data? Sensitivty value of calibration instrument needede y for yes, n for no: '))
            while Globals.do_test == 'error':
                print('Error: Input needs to be either y or n')
                Globals.do_loq=  generalUtilities.make_bool(input('Post modelling function: Calculate LOQ of each sample for each regression model, and then recalculate rmse and r2 statistics from the train and test prediction data? y for yes, n for no: '))
        if Globals.do_loq and Globals.sensitivity is None:
            Globals.sensitivity = float(input("Provide a float value for the sensitivty value of the calibration instrument, for loq calculations: "))
    def directories(data_folder = None, outpath = None, spectra_path = None, meta_path = None):
        if data_folder is None:
            data_folder, all_files = dataLoadingUtilities.get_data_folder()
            Globals.data_folder = data_folder
            Globals.all_files = all_files
        else:
            all_files = os.listdir(Globals.data_folder)
        if outpath is None:
            outpath = dataLoadingUtilities.get_out_folder()
            Globals.outpath = outpath
        # if spectra provided set globals, otherwise they're set to none
        if spectra_path is not None:
            path = os.path.join(Globals.data_folder, Globals.spectra_path)
            spectra = pd.read_csv(path, index_col=False)
            Globals.spectra = spectra
        else: 
            Globals.spectra_path = dataLoadingUtilities.get_spectra_path(Globals.data_folder)
        
        if meta_path is not None:
            Globals.meta_path = os.path.join(data_folder, meta_path)
            meta = pd.read_csv(Globals.meta_path, index_col = False)
            meta = meta.loc[:, ~meta.columns.str.contains('^Unnamed')]
            Globals.meta = meta
            print(Globals.meta)

        else:
            Globals.meta_path = dataLoadingUtilities.get_meta_path(Globals.data_folder)
    
    #input file directory
    def get_data_folder():
        in_prompt = 'Folder path containing data: '
        data_folder = input(in_prompt)
        while not os.path.exists(data_folder):
            #print(os.path.join(os.getcwd(), data_folder))
            print(f'Error: path {data_folder} does not exist\n')
            data_folder = input(in_prompt)
        all_files = os.listdir(data_folder)
        return data_folder, all_files

    #output file directory
    def get_out_folder():
        out_prompt = 'Folder path to export results: '
        outpath = input(out_prompt)
        while not os.path.exists(outpath):
            print(f'Error: path {outpath} does not exist\n')
            outpath = input(out_prompt)
        return outpath

    # get user-input spectra path, or sets to none if no spectra
    def get_spectra_path(data_folder):
        spectra_prompt = 'Spectra filename (hit ENTER for no spectra file): '
        ans = input(spectra_prompt)
        if not ans:
            Globals.spectra = None
            print("No spectra file recieved")
            return None
        else:
            spectra_file = dataLoadingUtilities.check_csv(ans)
            all_files = os.listdir(data_folder)
            #makes sure spectra input is good, adds it to globals spectra path and spectra datafames
            while spectra_file not in all_files:
                print(f'Error: file {spectra_file} not in data folder\n')
                spectra_file = dataLoadingUtilities.check_csv(input(spectra_prompt))
            while 'spectra' not in spectra_file.lower():
                print('word "spectra" missing from file name. Please make sure right file is supplied, and rename said file for your own sake')
                spectra_path = input(spectra_prompt)
            spectra_path = os.path.join(data_folder, spectra_file)
            Globals.spectra_path = spectra_path
            spectra = pd.read_csv(spectra_path, index_col=False)
            Globals.spectra = spectra
        return spectra_path

    # get user-input metadata path
    def get_meta_path(data_folder):
        meta_prompt = 'Metadata filename (hit ENTER for no meta file): '
        ans = input(meta_prompt)
        if not ans:
            Globals.meta = None
            print("No meta file supplied")
            return None
        else: 
            meta_file = dataLoadingUtilities.check_csv(ans)
            all_files = os.listdir(data_folder)
            while meta_file not in all_files:
                print(f'Error: file {meta_file} not in data folder\n')
                meta_file = dataLoadingUtilities.check_csv(input(meta_prompt))
            meta_path = os.path.join(data_folder, meta_file)
            Globals.meta = pd.read_csv(meta_path, index_col=False)
            return meta_path
    # check format of input .csv filename
    def check_csv(filename):
        if filename[-4:] != '.csv':
            filename = filename + '.csv'
        return filename  

#class for does the data input make sense? Anything to remove?
class metaUtilities:
    def generate_meta_statistics():
        print("Genreal statistics generated on metadata:")
        for var in Globals.var_to_run:
            print(f"Variable {var} :")
            print("\t Median", Globals.meta[f'{var}'].mean())
            print("\t Mean",Globals.meta[f'{var}'].median())
            print("\t Standard Deviation",statistics.stdev(Globals.meta[f'{var}']))
    def check_meta_format(meta, meta_path):
        var_to_run = [col for col in meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)]
        if len(var_to_run) == 0:
            raise ValueError('No variables identified in metadata')
        Globals.var_to_run = var_to_run
        all_var = ', '.join(var_to_run) 
        for var in var_to_run:
            if f'{var}_Folds' not in meta.columns:
                #runs entire stratification script, since user liekly doesn't have any fold columns.
                print('\n Bad metadata format, missing fold column for sample variable ' +  var + ' . Running stratification script on ENTIRE metadata file')
                Globals.meta = stratify_samples.stratification.main(meta_path = meta_path, var_list = var_to_run, n_folds = Globals.n_folds)
                if "sum" in Globals.var_to_run:
                    Globals.var_to_run.remove("sum")
                break
            #add uniue folds to dict for var, and filter out df for any -1 values among all folds
        return var_to_run, all_var, meta
    #replaces nan with -1, cleans out fold rows that are all -1. Should be called after confirming meta and spectra align for that reason
    def clean_meta_for_negatives(meta, var_to_run):
        Globals.meta = Globals.meta.fillna(-1)
        for i in range(len(Globals.meta)):
            flag = True 
            for var in var_to_run:
                #if at any point, a row of folds has a non-negative value, flag is marked false
                if Globals.meta.loc[i,f'{var}_Folds'] != -1:
                    flag = False
                    break
            #so if flag is still true, row must be removed
            if flag:
                Globals.meta.drop(index = meta.iloc[i].name, inplace = True)
                print("Removed index", i, "from meta file since fold values all negative")
        #export this new meta file in the original's place
        Globals.meta.to_csv(Globals.meta_path, index = False)
        print("Exported new cleaned metadata file at", Globals.meta_path)
       
class generalUtilities:
    def parser(sysargs = None):
        #REGRESSION STUFF
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

        #can it be a combo?
        normalization_prompt = '''Run normalizaotin on spectra data using:
            [0] SuperCam method
            [1] ChemCam method
            [2] SuperLIBS 10K method
            [3] SuperLIBS 18K method
        '''
        std_prompt = 'Should all variables follow the same modelling procedure? (set test fold, model type(s)) (y/n): '

        #POSSIBLE ARGUMENTS 
        # TO DO: REMOVE HIDE_PROGRESS
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--datafolder', type=str, default=None, help='Path of folder with data')
        parser.add_argument('-o', '--outpath', type=str, default=None, help='Path of folder to output results')
        parser.add_argument('-s', '--spectra_name', type=str, default=None, help='Spectra filename')
        parser.add_argument('-m', '--meta_name', type=str, default=None, help='Metadata filename')
        parser.add_argument('-std', '--standard', type=bool, default=None, help='Follow a standard procedure for each variable (bool)')
        parser.add_argument('-dt', '--do_test', type=bool, default=None, help='Hold a fold out as test data (bool)')
        parser.add_argument('-mt', '--method', type=str, default=None, help=f'Number corresponding to method selection from: {method_prompt}')
        parser.add_argument('-hp', '--hide_progress', type=bool, default=None, help='Hides progress bars')
        parser.add_argument('-mc', '--max_components', type=int, default=None, help='Sets the maximum PLS components')
        parser.add_argument('-np', '--num_params', type=int, default=None, help='Sets the number of values to test for LASSO, Ridge, ElasticNet, SVR')
        parser.add_argument('-pd', '--poly_deg', type=int, default=None, help='Sets the polynomial degree for SVR and kernel PCR')
        parser.add_argument('-mn', '--max_neighbors', type=int, default=None, help='Sets the maximum number of neighbors for kNN')
        #meta option:
        parser.add_argument('-n','--n_folds', type=int, default=None, help='Number of folds in case stratification script is called on metadata')
        #spectra options:
        parser.add_argument('-blr', '--baseline_removal', type=bool, default=None, help='Use baseline removal on spectra data?')
        parser.add_argument('--resample', type= int, default=None, help='Resample data to match [1] or to min step [2]')
        parser.add_argument('-norm', '--normalization', type=int, default=None, help=f'Normalize spectra data using method selection from: {normalization_prompt}')
        #below not used, but exists 
        parser.add_argument('--loq', type=bool, default=None, help='Post modelling function: Calculate LOQ of each sample for each regression model, and then recalculate rmse and r2 statistics from the train and test prediction data. y for yes, n for no: ')
        parser.add_argument('-sen', '--sensitivity', type=float, default=None, help='Float value for the sensitivty value of the calibration instrument,')
        args=parser.parse_args(sysargs)
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
        standard = args.standard
        Globals.standard = standard
        do_test = args.do_test
        if do_test is not None:
            Globals.do_test = do_test
        method_type = args.method
        if method_type is not None:
            method_type = method_type.replace("'","")
        #spectra args
        baseline_removal = args.baseline_removal
        if baseline_removal is not None:
            baseline_removal = baseline_removal.replace("'","")
            Globals.baseline_removal = baseline_removal
        resample = args.resample
        if resample is not None:
            resample = resample.replace("'","")
            Globals.resample= resample
            baseline_removal = args.baseline_removal
        normalization = args.normalization
        if normalization is not None:
            normalization  = normalization.replace("'","")
            Globals.normalization  = normalization 
        #meta arg
        n_folds = args.n_folds
        if n_folds  is not None:
            n_folds = n_folds.replace("'","")
            Globals.n_folds   = n_folds
        
        hide_progress = args.hide_progress
        max_components_ = args.max_components
        num_params_ = args.num_params
        poly_deg_ = args.poly_deg
        max_neighbors_ = args.max_neighbors

    def make_bool(val):
        if val not in ['y','n']:
            return 'error'
        if val == 'y':
            return True
        elif val == 'n':
            return False
    # select spectra from df and convert to array for modelling
    def select_spectra(spectra, sample_names):
        conv_spectra = np.array(spectra[sample_names].T)
        return conv_spectra
    # check if num is float (for coef plot)
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False
        
class foldUtilities:
    # identify relevant fold column
    def get_fold_col(var):

        if f'{var}_Folds' in Globals.meta.columns:
            fold_col = f'{var}_Folds'
        elif 'Folds' in Globals.meta.columns:
            fold_col = 'Folds'
        else:
            raise ValueError(f"Must either have an assigned '{var}_Folds' or general 'Folds' Pcolumn")

        return fold_col
    
    # identify the fold that is most like the full dataset
    # sets both non_test_folds_per_var and test_fold_per_var, where applicable
    #TESTING NEEDED FROM LIBS
    def get_most_representative_folds(do_plot=False):
        var_cols = [x.replace('_Folds','') for x in Globals.meta.columns if '_Folds' in x]
        # if multiple
        if len(var_cols)>1:
            print(f'Returning best fold column for all {len(var_cols)} identified variables')
        elif len(var_cols) == 0:
            raise ValueError('No fold columns identified. Have you stratified your metadata?')
    
        best_fold_dict = dict()
        for var in var_cols:
            assert var in Globals.meta.columns
            # get the folds
            all_folds = list(Globals.meta[f'{var}_Folds'].unique())
            # ignore the samples that aren't to be modelled
            if -1 in all_folds:
                all_folds.remove(-1)
            all_folds.sort()
            #only one fold, so test fold can't be done
            if (len(all_folds) == 1):
                Globals.non_test_folds_per_var[var] = all_folds
                Globals.test_fold_per_var = []
                continue 
            Globals.non_test_folds_per_var[var] = all_folds
            # get the histogram of all data, to match
            # choosing optimal number of bins by the Freedman-Diaconis rule
            hist_all, bin_edges = np.histogram(
                Globals.meta[
                    (Globals.meta[var].isna()) & # has a value
                    (Globals.meta[f'{var}_Folds']!=-1) # isn't an outlier
                ][var].values, 
                bins='fd'
            )
            hist_all_per = hist_all/sum(hist_all)
        
            if do_plot:
                # make colormap key evenly split
                cm_key = np.linspace(0,1,len(all_folds))
                # plot main data       
                fig,ax = plt.subplots(
                    nrows=len(all_folds)+1,
                    figsize=(6,14)
                )
                ax[0].bar(bin_edges[:-1], hist_all_per, width=np.diff(bin_edges), color='black')
                ax[0].set_title(f'Original {var} data')
        
            # get difference in histograms for each fold
            fold_diff_dict = dict()
            for i, fold in enumerate(all_folds):
                # get histogram using the same bins are for all the data
                fold_hist = np.histogram(
                    Globals.meta[Globals.meta[f'{var}_Folds']==fold][var].values, 
                    bins=bin_edges)[0]        
                # convert to percentages per bin
                fold_hist_per = fold_hist/sum(fold_hist)
                
                # calculate the mean abs difference to the full hist
                # https://stackoverflow.com/questions/50430585/mean-absolute-difference-of-two-numpy-arrays
                fold_diff = np.mean(np.abs(fold_hist_per[:,None] - hist_all_per))
                # add to dict
                fold_diff_dict[fold] = fold_diff
                
                # plot
                if do_plot:
                    ax[i+1].bar(bin_edges[:-1], 
                                fold_hist_per, 
                                width=np.diff(bin_edges),
                               color=cm.viridis(cm_key[i]))
                    ax[i+1].set_title(f'{var} Fold {fold}: {round(fold_diff*100,5)}% difference')
            if do_plot:
                # display plot
                plt.tight_layout()
                plt.show()
        
            # choose best fold, add to test fold dictionary
            best_fold = min(fold_diff_dict, key=fold_diff_dict.get)
            best_fold_dict[var] = best_fold
            Globals.test_fold_per_var[var] = best_fold
            # fill out temp meta, basically total data - test_fold_selected 
            fold_col = foldUtilities.get_fold_col(var)
            #if a test fold is selected, then add the values outside of it to the temp_meta dictionary and remove that test fold number from the non test fold numbers dictionary
            if best_fold is not None:
                Globals.temp_meta[var] = Globals.meta[
                    Globals.meta[fold_col] != best_fold
                    ].copy()
                #remove test folds from non test folds for variable
                Globals.non_test_folds_per_var[var].remove(best_fold)
            else:
                Globals.temp_meta[var] = Globals.meta[fold_col].copy()

        print("Test folds selection dictionary:", Globals.test_fold_per_var)
        print("Non-testfolds by sample:", Globals.non_test_folds_per_var)
        #not necessary, delete below?   
        # if only one variable, just return it
        # if len(var_cols) == 1:
        #     return best_fold
        # # otherwise, return dict
        # else:
        #     return best_fold_dict

    # convert data to dict of train/test dfs per fold for a variable (ONE VARIABLE INPUT)
    def make_data_dict(var, fold_col, test_fold = None):
        #get temp_meta[var] for what's being used and the folds to iterate through
        #folds_list = list(Globals.non_test_folds_per_var[var])
        #data = Globals.temp_meta[var]
        #print(data)
        data_dict = {}
        # to specify model params later
        n_samples_list = []
        #temp_meta is metadata - test_fold rows
        if test_fold is not None:
            #temp_meta = Globals.meta[Globals.meta[fold_col] != test_fold].copy()
            temp_meta = Globals.temp_meta[var]
        else:
            temp_meta = Globals.meta.copy()

        all_folds = list(Globals.non_test_folds_per_var[var]) #list(temp_meta[fold_col].unique())
        if -1 in all_folds:
            all_folds.remove(-1)
        
        #iterate through non-test folds for each sample to create train-test splits to add to data dictionary
        for fold in all_folds:
             #  training data, x_train selects train folds - current fold
            train_meta = temp_meta[(temp_meta[fold_col] != fold) &
                                   (temp_meta[fold_col] != -1)].reset_index(drop=True)
            X_train = generalUtilities.select_spectra(Globals.spectra, train_meta.pkey)
            y_train = train_meta[var].values
            n_samples_list.append(len(y_train))

           # held-out data, aS current fold becomes test
            test_meta = temp_meta[temp_meta[fold_col] == fold].reset_index(drop=True)
            X_test = generalUtilities.select_spectra(Globals.spectra, test_meta.pkey)
            y_test = test_meta[var].values 
            n_samples_list.append(len(y_test))

            # add datasets to dictionary
            data_dict[fold] = {'train_spectra':X_train,
                               'train_metadata':y_train,
                               'test_spectra':X_test,
                               'test_metadata':y_test}

        min_samples = min(n_samples_list)
        return data_dict, min_samples

        
class modelUtilities:
    # convert data to correct format for modelling
    # dump .asc files, and coeff files if applicable in outpath folders
    def format_spectra_meta( var, fold_col, test_fold=None):
        if test_fold is None:
            train_meta = Globals.meta[(Globals.meta[fold_col] != -1) &
                              (~Globals.meta[fold_col].isnull())]
            y_train = train_meta[var].values
            train_names = train_meta['pkey'].values
            X_train = generalUtilities.select_spectra(Globals.spectra, train_names)

            return train_names, X_train, y_train
        else:
            # training
            train_meta = Globals.meta[(~Globals.meta[fold_col].isin([-1, test_fold])) &
                              (~Globals.meta[fold_col].isnull())]
            y_train = train_meta[var].values
            train_names = train_meta['pkey'].values
            X_train = generalUtilities.select_spectra(Globals.spectra, train_names)

            # testing
            test_meta = Globals.meta[(Globals.meta[fold_col] == test_fold) &
                             (~Globals.meta[fold_col].isnull())]
            y_test = test_meta[var].values
            test_names = test_meta['pkey'].values
            X_test = generalUtilities.select_spectra(Globals.spectra, test_names)

            return train_names, X_train, y_train, test_names, X_test, y_test
    # for choosing best parameter during CV
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
# perform manual CV using data_dict and return RMSECV
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
            test_df = pd.DataFrame({
                'actual': y_test.flatten().tolist(),
                'pred' : preds.flatten().tolist()
            })
            rmsep = sqrt(mean_squared_error(test_df.actual, test_df.pred))
            rmsep_list.append(rmsep)
        rmsecv = mean(rmsep_list)
        return rmsecv
    
    #base pls function, taken from what's under the main loop of run_PLS, that is called by PLS parallel 
    def base_PLS(n_components):
        # define model
        model = PLSRegression(n_components = n_components, scale=False)
        # run CV and get RMSE
        temp_rmsecv = modelUtilities.run_CV( model)
        # add results to dictionary
    # cv_dict[temp_rmsecv] = n_components
        return model, temp_rmsecv, n_components

    #PLS in parallel
    #TO DO: progress bar better way?
    def PLS_parallel(max_components):
        # with Pool(1) as p:
        #     p.map(base_PLS, [2, 3, 4])
        cv_dict = {}
        out = Parallel(n_jobs=-1, verbose = 0, prefer = "threads")(delayed(modelUtilities.base_PLS)(n_components = components)  for components in tqdm(range(2,max_components +1))) #tdm(component_range, desc='component value')
        for components, values in enumerate(out):
            model, temp_rmsecv, n_components = values[0], values[1], values[2]
            cv_dict[temp_rmsecv] = n_components
        rmsecv = modelUtilities.get_first_local_minimum(list(cv_dict.keys()))
        component = cv_dict[rmsecv]
        model = PLSRegression(n_components = component, scale=False)

        if Globals.hide_progress is False:
            print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from {component}-component model')
        return component, rmsecv, model

    def run_PLS( max_components):
        
        component_range = np.arange(start=2, stop=max_components+1, step=1)

        cv_dict = {}
        for n_components in tqdm(component_range, desc='component value', disable=Globals.hide_progress):
            # define model
            model = PLSRegression(n_components = n_components, scale=False)
            # run CV and get RMSE
            temp_rmsecv = modelUtilities.run_CV( model)
            # add results to dictionary
            cv_dict[temp_rmsecv] = n_components
            
        rmsecv = modelUtilities.get_first_local_minimum(list(cv_dict.keys()))
        component = cv_dict[rmsecv]
        model = PLSRegression(n_components = component, scale=False)

        if Globals.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from {component}-component model')
            
        return component, rmsecv, model

    def run_LASSO(num_alphas):
        
        alpha_range = np.logspace(-10, 1, num_alphas)

        cv_dict = dict()
        for alpha in tqdm(alpha_range, desc='alpha value', disable=Globals.hide_progress):
            model = Lasso(alpha=alpha)
            temp_rmsecv = modelUtilities.run_CV(model)
            cv_dict[temp_rmsecv] = alpha

        rmsecv = modelUtilities.get_first_local_minimum(list(cv_dict.keys()))
        alpha = cv_dict[rmsecv]
        model = Lasso(alpha=alpha)
        
        if Globals.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(alpha,5)}')
            
        return alpha, rmsecv, model
            
    def run_Ridge( num_alphas):
        
        alpha_range = np.logspace(-10, 1, num_alphas)
        
        cv_dict = dict()
        for alpha in tqdm(alpha_range, desc='alpha value', disable=Globals.hide_progress):
            model = Ridge(alpha=alpha)
            temp_rmsecv = modelUtilities.run_CV( model)
            cv_dict[temp_rmsecv] = alpha
        
        rmsecv = modelUtilities.get_first_local_minimum(list(cv_dict.keys()))
        alpha = cv_dict[rmsecv]
        model = Ridge(alpha=alpha)
        
        if Globals.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(alpha,5)}')
            
        return alpha, rmsecv, model
    
    def run_ElasticNet( num_alphas):
        
        # suggested by documentation to skew to lasso
        ratio_range = [.1, .5, .7, .9, .95, .99, 1]
        # slightly raise min because takes longer
        alpha_range = np.logspace(-7, 1, num_alphas)

        cv_dict = dict()
        for ratio in tqdm(ratio_range, desc='L1 ratio', leave=False, disable=Globals.hide_progress):
            for alpha in tqdm(alpha_range, desc='alpha value', leave=False, disable=Globals.hide_progress):
                model = ElasticNet(alpha=alpha, l1_ratio=ratio)
                temp_rmsecv = modelUtilities.run_CV( model)
                cv_dict[temp_rmsecv] = [alpha, ratio]
        
        rmsecv = min(list(cv_dict.keys()))
        params = cv_dict[rmsecv]
        model = ElasticNet(alpha=params[0], l1_ratio=params[1])
        
        if Globals.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(params[0],5)} and an l1_ratio of {params[1]}')
        param = f'alpha={params[0]} l1_ratio={params[1]}'
            
        return param, rmsecv, model    
        
           
    def run_SVR_linear( num_epsilons):
        
        # smaller range here
        epsilon_range = np.logspace(-4, 1, num_epsilons)

        cv_dict = dict()
        for epsilon in tqdm(epsilon_range, desc='epsilon value', disable=Globals.hide_progress):
            model = SVR(kernel='linear', epsilon=epsilon)
            temp_rmsecv = modelUtilities.run_CV( model)
            cv_dict[temp_rmsecv] = epsilon
        
        rmsecv = min(list(cv_dict.keys()))
        epsilon = cv_dict[rmsecv]
        model = SVR(kernel='linear', epsilon=epsilon)
        
        if Globals.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an epsilon of {round(epsilon,5)}')
            
        return epsilon, rmsecv, model
    
    def run_SVR_poly( num_epsilons, poly_deg):
        
        print(f'Currently using a polynomial degree of {poly_deg}')
        
        epsilon_range = np.logspace(-4, 1, num_epsilons)

        cv_dict = dict()
        for epsilon in tqdm(epsilon_range, desc='epsilon value', disable=Globals.hide_progress):
            model = SVR(kernel='poly', degree=poly_deg, epsilon=epsilon)
            temp_rmsecv = modelUtilities.run_CV( model)
            cv_dict[temp_rmsecv] = epsilon
        
        rmsecv = min(list(cv_dict.keys()))
        epsilon = cv_dict[rmsecv]
        model = SVR(kernel='poly', degree=poly_deg, epsilon=epsilon)
        
        if Globals.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an epsilon of {round(epsilon,5)}')
            
        return epsilon, rmsecv, model
    
    def run_PCR_linear():
        
        print('PCR-lin does not optimize')
        # removed component range because different thing
        model = Pipeline([('PCA', PCA()), ('linear', LinearRegression())])
        rmsecv = modelUtilities.run_CV( model)
        
        if Globals.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
            
        return 'NA', rmsecv, model
    
    def run_PCR_poly( poly_deg):
        
        print('PCR-py does not optimize')
        #print(f'Currently using a polynomial degree of {poly_deg}')

        pca = KernelPCA(kernel='poly', degree=poly_deg)
        model = Pipeline([('PCA',pca), ('linear', LinearRegression())])
        rmsecv = modelUtilities.run_CV( model)
        
        if Globals.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
        
        return 'NA', rmsecv, model
    
    def run_OMP():
        
        print('OMP does not optimize')
        model = OrthogonalMatchingPursuit()
        rmsecv = modelUtilities.run_CV( model)
        
        if Globals.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
        
        return 'NA', rmsecv, model
    
    def run_RF():
        
        feat_range = ['sqrt', 'log2'] # `None` took long

        cv_dict = dict()
        for feat in tqdm(feat_range, desc='max features', disable=Globals.hide_progress):
            model = RandomForestRegressor(max_features=feat)
            temp_rmsecv = modelUtilities.run_CV( model)
            cv_dict[temp_rmsecv] = feat
        
        rmsecv = min(list(cv_dict.keys()))
        feat = cv_dict[rmsecv]
        model = RandomForestRegressor(max_features=feat)
        
        if Globals.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with {feat} max features')
            
        return feat, rmsecv, model
    
    def run_GBR():
        
        feat_range = ['sqrt', 'log2'] # `None` took long

        cv_dict = dict()
        for feat in tqdm(feat_range, desc='max features', disable=Globals.hide_progress):
            model = GradientBoostingRegressor(random_state=0, max_features=feat)
            temp_rmsecv = modelUtilities.run_CV( model)
            cv_dict[temp_rmsecv] = feat
        
        rmsecv = min(list(cv_dict.keys()))
        feat = cv_dict[rmsecv]
        model = GradientBoostingRegressor(random_state=0, max_features=feat)
        
        if Globals.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model with {feat} max features')
        
        return feat, rmsecv, model
    
    def run_OLS():
        
        print('OLS does not optimize')
        model = LinearRegression()
        rmsecv = modelUtilities.run_CV( model)
        
        if Globals.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
        
        return 'NA', rmsecv, model
    
    def run_kNN( max_neighbors):

        if max_neighbors > 1:
            neighbor_range = np.arange(1,max_neighbors)
        else:
            neighbor_range = [1]
            
        weight_range = ['uniform','distance']

        cv_dict = dict()
        for neighbor in tqdm(neighbor_range, desc='# neighbors', disable=Globals.hide_progress):
            for weight in weight_range:
                model = KNeighborsRegressor(n_neighbors=neighbor, weights=weight)
                temp_rmsecv = modelUtilities.run_CV( model)
                cv_dict[temp_rmsecv] = [neighbor, weight]
        
        rmsecv = min(list(cv_dict.keys()))
        params = cv_dict[rmsecv]
        model = KNeighborsRegressor(n_neighbors=params[0], weights=params[1])
        
        if Globals.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with {round(params[0],5)} neighbors and {params[1]} weights')
        param = f'n_neighbors={params[0]} weights={params[1]}'
        
        return param, rmsecv, model

#Plot functions for models.py
class plotUtilities():
    
    '''
    Functions that plot model information
    '''
    
    # model coefficients overlaid over example spectrum
    def coeffs(df, spectrum, var, method, path):
    
        # add example spectrum to df
        df['spectrum'] = spectrum    

        # check for and remove non-numeric channels
        all_n = len(df)
        df = df[df['wave'].map(generalUtilities.isfloat)].reset_index(drop=True)
        df['wave'] = df['wave'].astype(float)
        new_n = len(df)
        n_removed = all_n - new_n
        if n_removed > 0:
            print(f'{n_removed} non-numeric channels removed from coefficient plot')

        # PARAMETERS
        color1 = 'black' # spectrum
        color2 = '#e1dd01' # coeffs fill
        color3 = '#e07b00' # coeffs outline + text

        fsize = 14 # font size
        dsize = 30 # dot size

        opacity1 = 0.6 # spectrum
        opacity2 = 0.8 # coeffs

        # drop 0s
        coef_df = df[df['coef']!=0].reset_index(drop=True)

        # make plot
        fig, ax = plt.subplots(figsize=(10,5))

        # spectrum
        ax.plot(df['wave'], 
                df['spectrum'],
                color=color1, 
                lw=2, 
                alpha=opacity1)
        # coefficients
        ax2 = ax.twinx()
        ax2.scatter(coef_df['wave'], 
                    coef_df['coef'], 
                    alpha = opacity2, 
                    c=color2, 
                    marker='o',
                    edgecolors=color3,
                    s=dsize)

        ax.set_xlabel('Channel', fontsize=fsize)
        ax.set_ylabel('Intensity', fontsize=fsize)
        ax2.set_ylabel('Coefficient Weight', fontsize=fsize, color = color3)
        plt.title(var, size=fsize+2)
        
        # save plot
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'{var}_{method}_coefs_plot.jpg'), dpi=600)
        plt.savefig(os.path.join(path, f'{var}_{method}_coefs_plot.eps'), dpi=600)
        plt.close()
        
    # predicted vs true scatter plot with 1:1 line
    def pred_true(df, var, method, type, rmse, adj_r2, path):
        
        actual_col = f'{var}_actual'
        pred_col = f'{var}_pred'

        # PARAMETERS
        size=14 # font size
        color = 'black' #'#3f997c' # dot color
        opacity = 0.6 # dot opacity

        # get plot limits for 1:1 line
        plot_dict = plotUtilities.get_limits_for_1_to_1_line(df, actual_col, pred_col)

        # make plot
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(df[actual_col], df[pred_col], c=color, alpha=opacity)
        ax.plot(
            [plot_dict['plt_min'],plot_dict['plt_max']], 
            [plot_dict['plt_min'],plot_dict['plt_max']],
            'k--')
        ax.plot(
            [plot_dict['plt_min'],plot_dict['plt_max']], 
            [0,0], 
            'k')
        plt.xlim(plot_dict['x_min'],plot_dict['x_max'])
        plt.ylim(plot_dict['y_min'], plot_dict['y_max'])

        plt.title(f'{type} RMSE: {round(rmse, 2)}    Adj. R2: {round(adj_r2, 2)}', fontsize=size)
        ax.set_ylabel(f'Predicted {var}', fontsize=size)
        ax.set_xlabel(f'Actual {var}', fontsize=size)

        # save plot
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'{var}_{method}_{type}_pred_true_plot.jpg'), dpi=600)
        plt.savefig(os.path.join(path, f'{var}_{method}_{type}_pred_true_plot.eps'), dpi=600)
        plt.close()
    
    #used in plot coefficients in plotUtiltiies for 1:1 line plot limits for plot_dict
    #get values for 1:1 line and then plot limits to refocus on data
    def get_limits_for_1_to_1_line(df, actual_col, pred_col):
        # values for 1:1 line
        plt_max = plotUtilities.get_max(max(max(df[actual_col].values), max(df[pred_col].values)))       
        plt_min = plotUtilities.get_min(min(min(df[actual_col].values), min(df[pred_col].values)))
        # get X plot limits
        x_min = plotUtilities.get_min(min(df[actual_col].values))
        x_max = plotUtilities.get_max(max(df[actual_col].values))
        # get y plot limits
        y_min = plotUtilities.get_min(min(df[pred_col].values))
        y_max = plotUtilities.get_max(max(df[pred_col].values))
        return {
            'plt_max':plt_max, 
            'plt_min':plt_min, 
            'x_min':x_min,
            'x_max':x_max,
            'y_min':y_min,
            'y_max':y_max
        }
    
    # get min axis value for plotting
    def get_min(value, buffer=0.1):
        if value > 0:
            value = 0
        else:
            value = value - (buffer * value)
        return value

    # get max axis value for plotting
    def get_max(value, buffer=0.1):
        if value > 0:
            value = value + (buffer * value)
        else:
            value = 0
        return value

    