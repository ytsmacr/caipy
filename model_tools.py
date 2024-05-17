# standard packages
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from statistics import mean, median
from tqdm import tqdm
import os, numbers
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
plt.set_loglevel('error')

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

'''
by Cai Ytsma (cai@caiconsulting.co.uk)
Last updated 17 May 2024

Standalone functions and classes used by other programs in caipy.

TODO: maybe these classes should be split into different .py files, and model_tools disintegrated?
'''

########################
# STANDALONE FUNCTIONS #
########################
# TODO: should these be added to the Analyze class? Will have to troubleshoot calls from other classes

# check format of input .asc filename
def check_asc(filename):
    if filename[-4:] != '.asc':
        filename = filename + '.asc'

    return filename

# check format of input .csv filename
def check_csv(filename):
    if filename[-4:] != '.csv':
        filename = filename + '.csv'

    return filename  

# convert y/n response to boolean
def make_bool(val):
    if val not in ['y','n']:
        return 'error'
    if val == 'y':
        return True
    elif val == 'n':
        return False

# check if num is float (for coef plot)
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
    
# convert spectra df to np array for modelling
def convert_spectra(spectra):
    first_col = spectra.columns[0]
    if first_col != 'wave':
        cont = make_bool(input(f'Warning: convert_spectra assumes the first column is the wavelength axis and ignores it. The first column of your data is {first_col}. Continue? (y/n):'))
        if not cont:
            raise ValueError('Aborting')

    conv_spectra = np.array(spectra[spectra.columns[1:]].T)
    return conv_spectra

# select spectra from df and convert to array for modelling
def select_spectra(spectra, sample_names):
    conv_spectra = np.array(spectra[sample_names].T)
    return conv_spectra

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

# get values for 1:1 line and then plot limits to refocus on data
def get_limits_for_1_to_1_line(df, actual_col, pred_col):
    # values for 1:1 line
    plt_max = get_max(max(max(df[actual_col].values), max(df[pred_col].values)))       
    plt_min = get_min(min(min(df[actual_col].values), min(df[pred_col].values)))
    # get X plot limits
    x_min = get_min(min(df[actual_col].values))
    x_max = get_max(max(df[actual_col].values))
    # get y plot limits
    y_min = get_min(min(df[pred_col].values))
    y_max = get_max(max(df[pred_col].values))
    return {
        'plt_max':plt_max, 
        'plt_min':plt_min, 
        'x_min':x_min,
        'x_max':x_max,
        'y_min':y_min,
        'y_max':y_max
    }

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

# get user-input data folder
def get_data_folder():
    in_prompt = 'Folder path containing data: '
    data_folder = input(in_prompt)
    while not os.path.exists(data_folder):
        print(f'Error: path {data_folder} does not exist\n')
        data_folder = input(in_prompt)
    all_files = os.listdir(data_folder)
    return data_folder, all_files

# get user-input spectra path
def get_spectra_path(data_folder, all_files):
    spectra_prompt = 'Spectra filename: '
    spectra_file = check_csv(input(spectra_prompt))
    while spectra_file not in all_files:
        print(f'Error: file {spectra_file} not in data folder\n')
        spectra_file = check_csv(input(spectra_prompt))
    spectra_path = os.path.join(data_folder, spectra_file)
    return spectra_path

# get user-input metadata path
def get_meta_path(data_folder, all_files):
    meta_prompt = 'Metadata filename: '
    meta_file = check_csv(input(meta_prompt))
    while meta_file not in all_files:
        print(f'Error: file {meta_file} not in data folder\n')
        meta_file = check_csv(input(meta_prompt))
    meta_path = os.path.join(data_folder, meta_file)
    return meta_path

# get user-input output folder
def get_out_folder():
    out_prompt = 'Folder path to export results: '
    outpath = input(out_prompt)
    while not os.path.exists(outpath):
        print(f'Error: path {outpath} does not exist\n')
        outpath = input(out_prompt)
    return outpath

##########################################

class Analyze():

    '''
    Standalone functions for commonly used analyses
    '''

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
        train_median = round(median(train_df[true].values),2)
        # remove those below LOQ
        train_above_loq = train_df[train_df[pred]>loq].copy()
        # recalculate metrics
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
            result_dict = {
                'median_conc_train':train_median,
                'median_conc_test':test_median,
                'loq':loq,
                'n_train_above_loq':rev_n_train,
                'median_conc_train_above_loq':rev_train_median,
                'rmsec_above_loq':rev_rmsec,
                'r2_train_above_loq':rev_train_r2,
                'adj_r2_train_above_loq':rev_train_adj_r2,
                'n_test_above_loq':rev_n_test,
                'median_conc_test_above_loq':rev_test_median,
                'rmsep_above_loq':rev_rmsep,
                'r2_test_above_loq':rev_test_r2,
                'adj_r2_test_above_loq':rev_test_adj_r2
            }
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
        
        # finally, export
        return result_dict

    def calculate_median_mean_sensitivity_plot(
            sensitivity_list: list, # list of sensitivity values
            descriptor: str, # name for plot
            plot_folder: str, # file path to export plot to
            make_plot = True,
            show_plot = True
            ):
        
        # drop na if there
        sensitivity_list = [x for x in sensitivity_list if not pd.isna(x)]

        # median
        median_sens = round(median(sensitivity_list),9)
        #print("Median sensitivity:", round(median_sens,1))

        # mean
        mean_sens = mean(sensitivity_list)
        #print("Mean sensitivity:", round(mean_sens,1))

        # compare median to mean
        med_c = 'red'
        mean_c = 'blue'

        if make_plot:
            # make plot of distributions
            plt.hist(sensitivity_list, bins=20)
            plt.ylabel("# Standards")
            plt.xlabel("Sensitivity")
            y_bot, y_top = plt.ylim()
            plt.vlines(x=median_sens,
                    ymin = 0,
                    ymax = y_top,
                    colors=med_c,
                    label='median')
            plt.vlines(x=mean_sens,
                    ymin = 0,
                    ymax = y_top,
                    colors=mean_c,
                    label='mean')
            plt.title(descriptor)
            plt.legend()
            plt.ylim((0,y_top))
            plt.savefig(os.path.join(plot_folder,f'{descriptor}_sensitivity_plot.jpg'), dpi=600)
            plt.savefig(os.path.join(plot_folder,f'{descriptor}_sensitivity_plot.eps'), dpi=600)
            if show_plot:
                plt.show()
            else:
                plt.close()

        # return values
        return median_sens, mean_sens


##########################################

class Preprocess():
    
    '''
    Standalone functions used to preprocess spectra

       
    RESAMPLE SPECTRA
    '''

    # resample spectra to given axis
    @staticmethod
    def resample_to_match(spectra, spectra_to_match = None):

        spectra_to_resample = spectra.copy()
        
        # using Spectres package
        ## https://spectres.readthedocs.io/en/latest/
        ## https://arxiv.org/pdf/1705.05165v1.pdf

        if spectra_to_match is None:
            print('Resampling to SuperCam -2px shifted wavelength axis by default')
            spectra_to_match = pd.read_csv(os.path.join('data','SuperCam_cal_shift-2pix_axis.csv'))

        # if input is a dataframe
        if isinstance(spectra_to_match,pd.DataFrame):
            if 'wave' not in spectra_to_match.columns:
                print('Axis to match needs "wave" header')
                return
            new_axis = spectra_to_match['wave'].to_numpy()

        # if input is an array
        elif isinstance(spectra_to_match,np.ndarray):
            new_axis = spectra_to_match

        # if input is an array
        elif isinstance(spectra_to_match,list):
            new_axis = np.array(spectra_to_match)

        else:
            print('Spectra to match must either be a dataframe with "wave" column, or a list of values')
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

    # resample uniformly to minimum step size
    @staticmethod
    def resample_to_min_step(spectra):
        
        spectra_to_resample = spectra.copy()

        if 'wave' not in spectra_to_resample.columns:
            print('Input spectra must have "wave" axis column')
            return

        axis = spectra_to_resample['wave'].to_numpy()

        # get step sizes
        step_set = set()
        for i in np.arange(len(wave))[:-1]:
            current = wave[i]
            next = wave[i+1]
            step = next-current
            step_set.add(step)

        # get minimum
        min_step = min(step_set)

        # populate new axis with this step size
        min_step_axis = np.arange(start = axis[0], stop = axis[-1]+min_step, step=min_step)

        # resample spectra to match this
        resampled_spectra = Preprocess.resample_to_match(spectra_to_resample, min_step_axis)
        return resampled_spectra

    '''
    BASELINE REMOVAL
    '''

    # airPLS baseline removal
    ## recommend that resample to min step size first ##
    @staticmethod
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

    @staticmethod
    def band_normalize(spectra:pd.DataFrame, band):
    # normalize the spectra to the value at a certain channel

        # check that numeric
        assert isinstance(band, numbers.Real)    

        # check that within bounds of axis
        src_axis = list(spectra['wave'])
        try:
            assert ((band > min(src_axis)) and (band < max(src_axis)))
        except AssertionError:
            print(f'Band location not within spectrum bounds ({min(src_axis)} - {max(src_axis)})')

        # GET VALUE AT BAND LOCATION
        needs_resampling = False
        new_axis = src_axis.copy()
        src_spectra = spectra.copy()

        # find closest value
        closest_wave = min(src_axis, key=lambda x:abs(x-band))
        closest_i = src_axis.index(closest_wave)
        
        # see if needs to go before
        if closest_wave > band:
            needs_resampling = True
            # add before (will replace index so stay the same)
            new_axis.insert(closest_i, band)

        elif closest_wave < band:
            needs_resampling = True
            # add after
            new_axis.insert(closest_i+1, band)

        # resample to add the band location value if needed
        if needs_resampling:
            new_spectra = Preprocess().resample_to_match(src_spectra, new_axis)
        else:
            new_spectra = src_spectra.copy()

        # get presentation of the band in the axis
        band = min(new_axis, key=lambda x:abs(x-band))

        # assign value to normalize to
        band_vals = new_spectra.set_index('wave').loc[band].values
        new_spectra = new_spectra.set_index('wave').T
        new_spectra.insert(0,'band_value',band_vals)

        # normalize
        normed_spectra = (
            new_spectra
            .div(new_spectra['band_value'], axis=0) # normalize
            .T # restructure
            .drop(index='band_value') # remove reference col
        ) 

        # remove the added row by resampling if necessary
        if needs_resampling:
            normed_spectra.drop(index=band, inplace=True)

        # reset and check matches input
        normed_spectra.reset_index(inplace=True)
        assert list(normed_spectra['wave']) == list(src_spectra['wave'])

        # extract the band value per spectrum
        return normed_spectra


    # normalize each df subset of data, then concatenate
    @staticmethod
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
    @staticmethod
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

        normed_spectra = Preprocess.normalize_regions(df_list)

        return normed_spectra

    # normalize by ChemCam method
    @staticmethod
    def norm3_CL(spectra):
        
        spectra_tonorm = spectra.copy()

        uv = spectra_tonorm[(spectra_tonorm['wave'] >= 246.68) & (spectra_tonorm['wave'] <= 338.42)].copy()
        vis = spectra_tonorm[(spectra_tonorm['wave'] >= 387.9) & (spectra_tonorm['wave'] <= 469.1)].copy()
        vnir = spectra_tonorm[(spectra_tonorm['wave'] >= 492.65) & (spectra_tonorm['wave'] <= 849.1)].copy()

        df_list = [uv, vis, vnir]

        normed_spectra = Preprocess.normalize_regions(df_list)

        return normed_spectra

    # normalize by SuperLIBS 10K method
    @staticmethod
    def norm3_SL_10K(spectra):
        
        spectra_tonorm = spectra.copy()

        uv = spectra_tonorm[(spectra_tonorm['wave'] >= 233.12) & (spectra_tonorm['wave'] <= 351.35)].copy()
        vis = spectra_tonorm[(spectra_tonorm['wave'] >= 370.16) & (spectra_tonorm['wave'] <= 479.07)].copy()
        vnir = spectra_tonorm[(spectra_tonorm['wave'] >= 498.14) & (spectra_tonorm['wave'] <= 859.44)].copy()

        df_list = [uv, vis, vnir]

        normed_spectra = Preprocess.normalize_regions(df_list)

        return normed_spectra

    # normalize by SuperLIBS 18K method
    @staticmethod
    def norm3_SL_18K(spectra):
        
        spectra_tonorm = spectra.copy()

        uv = spectra_tonorm[(spectra_tonorm['wave'] >= 233.12) & (spectra_tonorm['wave'] <= 351.35)].copy()
        vis = spectra_tonorm[(spectra_tonorm['wave'] >= 370.16) & (spectra_tonorm['wave'] <= 479.07)].copy()
        vnir = spectra_tonorm[(spectra_tonorm['wave'] >= 508.3) & (spectra_tonorm['wave'] <= 869.2)].copy()

        df_list = [uv, vis, vnir]

        normed_spectra = Preprocess.normalize_regions(df_list)

        return normed_spectra

##########################################

class Format():
    
    '''
    Functions used to prepare input data for modelling
    '''

    def __init__(self, spectra, meta):
        
        self.spectra = spectra
        self.meta = meta

    # identify relevant fold column
    def get_fold_col(self, var):

        if f'{var}_Folds' in self.meta.columns:
            fold_col = f'{var}_Folds'
        elif 'Folds' in self.meta.columns:
            fold_col = 'Folds'
        else:
            raise ValueError(f"Must either have an assigned '{var}_Folds' or general 'Folds' column")

        return fold_col
    
    # identify the fold that is most like the full dataset
    def get_most_representative_fold(self, variable=None, do_plot=False):

        if variable is None:
            # identify variable cols
            var_cols = [x.replace('_Folds','') for x in self.meta.columns if '_Folds' in x]
        else:
            # manual inputs can either be list (many) or string (one)
            if type(variable) == str:
                var_cols = [variable]
            elif type(variable) == list:
                var_cols = variable
            else:
                raise ValueError('Variable input must either be a string or list')
        
        # if multiple
        if len(var_cols)>1:
            print(f'Returning best fold column for all {len(var_cols)} identified variables')
        elif len(var_cols) == 0:
            raise ValueError('No fold columns identified. Have you stratified your metadata?')
    
        best_fold_dict = dict()
        for var in var_cols:
            assert var in self.meta.columns
            # get the folds
            all_folds = list(self.meta[f'{var}_Folds'].unique())
            # ignore the samples that aren't to be modelled
            if -1 in all_folds:
                all_folds.remove(-1)
            all_folds.sort()
            
            # get the histogram of all data, to match
            # choosing optimal number of bins by the Freedman-Diaconis rule
            hist_all, bin_edges = np.histogram(
                self.meta[
                    (~self.meta[var].isna()) & # has a value
                    (self.meta[f'{var}_Folds']!=-1) # isn't an outlier
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
                    self.meta[self.meta[f'{var}_Folds']==fold][var].values, 
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
        
            # choose best fold
            best_fold = min(fold_diff_dict, key=fold_diff_dict.get)
            best_fold_dict[var] = best_fold
    
        # if only one variable, just return it
        if len(var_cols) == 1:
            return best_fold
        # otherwise, return dict
        else:
            return best_fold_dict

    # convert data to dict of train/test dfs per fold
    def make_data_dict(self, var, fold_col, test_fold=None):
        
        # find minimum number of samples in the folds, 
        # to specify model params later
        n_samples_list = []

        # remove test data if using it
        if test_fold is not None:
            temp_meta = self.meta[self.meta[fold_col] != test_fold].copy()
        else:
            temp_meta = self.meta.copy()

        all_folds = list(temp_meta[fold_col].unique())
        if -1 in all_folds:
            all_folds.remove(-1)

        data_dict = {}
        for fold in all_folds:

            # training data
            train_meta = temp_meta[(temp_meta[fold_col] != fold) &
                                   (temp_meta[fold_col] != -1)].reset_index(drop=True)
            X_train = select_spectra(self.spectra, train_meta.pkey)
            y_train = train_meta[var].values
            n_samples_list.append(len(y_train))

            # held-out data
            test_meta = temp_meta[temp_meta[fold_col] == fold].reset_index(drop=True)
            X_test = select_spectra(self.spectra, test_meta.pkey)
            y_test = test_meta[var].values
            n_samples_list.append(len(y_test))

            # add datasets to dictionary
            data_dict[fold] = {'train_spectra':X_train,
                               'train_metadata':y_train,
                               'test_spectra':X_test,
                               'test_metadata':y_test}

        min_samples = min(n_samples_list)
        return data_dict, min_samples

    # convert data to correct format for modelling
    def format_spectra_meta(self, var, fold_col, test_fold=None):
        if test_fold is None:
            train_meta = self.meta[(self.meta[fold_col] != -1) &
                              (~self.meta[fold_col].isnull())]
            y_train = train_meta[var].values
            train_names = train_meta['pkey'].values
            X_train = select_spectra(self.spectra, train_names)

            return train_names, X_train, y_train
        else:
            # training
            train_meta = self.meta[(~self.meta[fold_col].isin([-1, test_fold])) &
                              (~self.meta[fold_col].isnull())]
            y_train = train_meta[var].values
            train_names = train_meta['pkey'].values
            X_train = select_spectra(self.spectra, train_names)

            # testing
            test_meta = self.meta[(self.meta[fold_col] == test_fold) &
                             (~self.meta[fold_col].isnull())]
            y_test = test_meta[var].values
            test_names = test_meta['pkey'].values
            X_test = select_spectra(self.spectra, test_names)

            return train_names, X_train, y_train, test_names, X_test, y_test

##########################################        

class Model():
    
    '''
    Functions that optimize regression models
    '''
    
    def __init__(self, data_dict, hide_progress=False):
        self.data_dict = data_dict
        self.hide_progress = hide_progress
            
    # perform manual CV using data_dict and return RMSECV
    def run_CV(self, model):

        rmsep_list = []
        for fold in list(self.data_dict.keys()):

            # get data
            X_train = self.data_dict[fold]['train_spectra']
            X_test = self.data_dict[fold]['test_spectra']
            y_train = self.data_dict[fold]['train_metadata']
            y_test = self.data_dict[fold]['test_metadata']

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
    
    # CV-optimize, return best model info
    def run_PLS(self, max_components):
        
        component_range = np.arange(start=2, stop=max_components+1, step=1)

        cv_dict = {}
        for n_components in tqdm(component_range, desc='component value', disable=self.hide_progress):
            # define model
            model = PLSRegression(n_components = n_components, scale=False)
            # run CV and get RMSE
            temp_rmsecv = Model.run_CV(self, model)
            # add results to dictionary
            cv_dict[temp_rmsecv] = n_components
            
        rmsecv = get_first_local_minimum(list(cv_dict.keys()))
        component = cv_dict[rmsecv]
        model = PLSRegression(n_components = component, scale=False)

        if self.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from {component}-component model')
            
        return component, rmsecv, model

    def run_LASSO(self, num_alphas):
        
        alpha_range = np.logspace(-10, 1, num_alphas)

        cv_dict = dict()
        for alpha in tqdm(alpha_range, desc='alpha value', disable=self.hide_progress):
            model = Lasso(alpha=alpha)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = alpha

        rmsecv = get_first_local_minimum(list(cv_dict.keys()))
        alpha = cv_dict[rmsecv]
        model = Lasso(alpha=alpha)
        
        if self.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(alpha,5)}')
            
        return alpha, rmsecv, model
            
    def run_Ridge(self, num_alphas):
        
        alpha_range = np.logspace(-10, 1, num_alphas)
        
        cv_dict = dict()
        for alpha in tqdm(alpha_range, desc='alpha value', disable=self.hide_progress):
            model = Ridge(alpha=alpha)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = alpha
        
        rmsecv = get_first_local_minimum(list(cv_dict.keys()))
        alpha = cv_dict[rmsecv]
        model = Ridge(alpha=alpha)
        
        if self.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(alpha,5)}')
            
        return alpha, rmsecv, model
    
    def run_ElasticNet(self, num_alphas):
        
        # suggested by documentation to skew to lasso
        ratio_range = [.1, .5, .7, .9, .95, .99, 1]
        # slightly raise min because takes longer
        alpha_range = np.logspace(-7, 1, num_alphas)

        cv_dict = dict()
        for ratio in tqdm(ratio_range, desc='L1 ratio', leave=False, disable=self.hide_progress):
            for alpha in tqdm(alpha_range, desc='alpha value', leave=False, disable=self.hide_progress):
                model = ElasticNet(alpha=alpha, l1_ratio=ratio)
                temp_rmsecv = Model.run_CV(self, model)
                cv_dict[temp_rmsecv] = [alpha, ratio]
        
        rmsecv = min(list(cv_dict.keys()))
        params = cv_dict[rmsecv]
        model = ElasticNet(alpha=params[0], l1_ratio=params[1])
        
        if self.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(params[0],5)} and an l1_ratio of {params[1]}')
        param = f'alpha={params[0]} l1_ratio={params[1]}'
            
        return param, rmsecv, model    
        
           
    def run_SVR_linear(self, num_epsilons):
        
        # smaller range here
        epsilon_range = np.logspace(-4, 1, num_epsilons)

        cv_dict = dict()
        for epsilon in tqdm(epsilon_range, desc='epsilon value', disable=self.hide_progress):
            model = SVR(kernel='linear', epsilon=epsilon)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = epsilon
        
        rmsecv = min(list(cv_dict.keys()))
        epsilon = cv_dict[rmsecv]
        model = SVR(kernel='linear', epsilon=epsilon)
        
        if self.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an epsilon of {round(epsilon,5)}')
            
        return epsilon, rmsecv, model
    
    def run_SVR_poly(self, num_epsilons, poly_deg):
        
        print(f'Currently using a polynomial degree of {poly_deg}')
        
        epsilon_range = np.logspace(-4, 1, num_epsilons)

        cv_dict = dict()
        for epsilon in tqdm(epsilon_range, desc='epsilon value', disable=self.hide_progress):
            model = SVR(kernel='poly', degree=poly_deg, epsilon=epsilon)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = epsilon
        
        rmsecv = min(list(cv_dict.keys()))
        epsilon = cv_dict[rmsecv]
        model = SVR(kernel='poly', degree=poly_deg, epsilon=epsilon)
        
        if self.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an epsilon of {round(epsilon,5)}')
            
        return epsilon, rmsecv, model
    
    def run_PCR_linear(self):
        
        print('PCR-lin does not optimize')
        # removed component range because different thing
        model = Pipeline([('PCA', PCA()), ('linear', LinearRegression())])
        rmsecv = Model.run_CV(self, model)
        
        if self.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
            
        return 'NA', rmsecv, model
    
    def run_PCR_poly(self, poly_deg):
        
        print('PCR-py does not optimize')
        #print(f'Currently using a polynomial degree of {poly_deg}')

        pca = KernelPCA(kernel='poly', degree=poly_deg)
        model = Pipeline([('PCA',pca), ('linear', LinearRegression())])
        rmsecv = Model.run_CV(self, model)
        
        if self.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
        
        return 'NA', rmsecv, model
    
    def run_OMP(self):
        
        print('OMP does not optimize')
        model = OrthogonalMatchingPursuit()
        rmsecv = Model.run_CV(self, model)
        
        if self.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
        
        return 'NA', rmsecv, model
    
    def run_RF(self):
        
        feat_range = ['sqrt', 'log2'] # `None` took long

        cv_dict = dict()
        for feat in tqdm(feat_range, desc='max features', disable=self.hide_progress):
            model = RandomForestRegressor(max_features=feat)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = feat
        
        rmsecv = min(list(cv_dict.keys()))
        feat = cv_dict[rmsecv]
        model = RandomForestRegressor(max_features=feat)
        
        if self.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with {feat} max features')
            
        return feat, rmsecv, model
    
    def run_GBR(self):
        
        feat_range = ['sqrt', 'log2'] # `None` took long

        cv_dict = dict()
        for feat in tqdm(feat_range, desc='max features', disable=self.hide_progress):
            model = GradientBoostingRegressor(random_state=0, max_features=feat)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = feat
        
        rmsecv = min(list(cv_dict.keys()))
        feat = cv_dict[rmsecv]
        model = GradientBoostingRegressor(random_state=0, max_features=feat)
        
        if self.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model with {feat} max features')
        
        return feat, rmsecv, model
    
    def run_OLS(self):
        
        print('OLS does not optimize')
        model = LinearRegression()
        rmsecv = Model.run_CV(self, model)
        
        if self.hide_progress is False:
           print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
        
        return 'NA', rmsecv, model
    
    def run_kNN(self, max_neighbors):

        if max_neighbors > 1:
            neighbor_range = np.arange(1,max_neighbors)
        else:
            neighbor_range = [1]
            
        weight_range = ['uniform','distance']

        cv_dict = dict()
        for neighbor in tqdm(neighbor_range, desc='# neighbors', disable=self.hide_progress):
            for weight in weight_range:
                model = KNeighborsRegressor(n_neighbors=neighbor, weights=weight)
                temp_rmsecv = Model.run_CV(self, model)
                cv_dict[temp_rmsecv] = [neighbor, weight]
        
        rmsecv = min(list(cv_dict.keys()))
        params = cv_dict[rmsecv]
        model = KNeighborsRegressor(n_neighbors=params[0], weights=params[1])
        
        if self.hide_progress is False:
           print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with {round(params[0],5)} neighbors and {params[1]} weights')
        param = f'n_neighbors={params[0]} weights={params[1]}'
        
        return param, rmsecv, model
    
##########################################   
    
class Plot():
    
    '''
    Functions that plot model information
    '''
    
    # model coefficients overlaid over example spectrum
    def coeffs(df, spectrum, var, method, path):
    
        # add example spectrum to df
        df['spectrum'] = spectrum    

        # check for and remove non-numeric channels
        all_n = len(df)
        df = df[df['wave'].map(isfloat)].reset_index(drop=True)
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
        plot_dict = get_limits_for_1_to_1_line(df, actual_col, pred_col)

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
