# standard packages
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from statistics import mean
from tqdm import tqdm
import os
import matplotlib.pyplot as plt 
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
Last updated 06 October 2022

Helper functions and classes used by other programs in auto-modelling.
'''

########################
# STANDALONE FUNCTIONS #
########################
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
    
    spec_list = []
    for column in spectra.columns[1:]:
        spectrum = list(spectra[column])
        spec_list.append(spectrum)

    conv_spectra = np.array(spec_list)

    return conv_spectra

# select spectra from df and convert to array for modelling
def select_spectra(spectra, sample_names):

    spec_list = []
    for column in sample_names:
        spectrum = list(spectra[column])
        spec_list.append(spectrum)

    conv_spectra = np.array(spec_list)

    return conv_spectra

# get min axis value for plotting
def get_min(value, buffer=0.1):

    if value > 0:
        value = 0
    else:
        value = value + (buffer * value)
    
    return value

# get max axis value for plotting
def get_max(value, buffer=0.1):

    if value > 0:
        value = value + (buffer * value)
    else:
        value = 0

    return value

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

class Preprocess():
    
    '''
    Functions used to preprocess spectra
    '''
    
    class Resample():
        
        '''
        RESAMPLE SPECTRA
        '''
    
        # resample spectra to given axis
        def resample_to_match(spectra_to_resample, spectra_to_match = None):

            # using Spectres package
            ## https://spectres.readthedocs.io/en/latest/
            ## https://arxiv.org/pdf/1705.05165v1.pdf

            if spectra_to_match is None:
                print('Resampling to SuperCam -2px shifted wavelength axis by default')
                spectra_to_match = pd.read_csv('data\\SuperCam_cal_shift-2pix_axis.csv')

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

            temp = spectra_to_resample.copy()
            old_axis = temp['wave'].to_numpy()

            if list(new_axis) == list(old_axis):
                print('Spectral axes already matched')
                return

            temp.drop('wave', axis=1, inplace=True)
            old_spectra = temp.T.to_numpy()

            new_spectra = spectres(new_axis, old_axis, old_spectra, fill=0, verbose=False)
            new_spectra = pd.DataFrame(new_spectra).T
            new_spectra.columns = temp.columns
            new_spectra.insert(0,'wave',new_axis)

            return new_spectra

        # resample uniformly to minimum step size
        def resample_to_min_step(spectra):

            if 'wave' not in spectra.columns:
                print('Input spectra must have "wave" axis column')
                return

            axis = spectra['wave'].to_numpy()

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
            resampled_spectra = Resample.resample_to_match(spectra, min_step_axis)
            return resampled_spectra
    
    class BLR():
        
        '''
        BASELINE REMOVAL
        '''
    
        # airPLS baseline removal
        ## recommend that resample to min step size first ##
        def AirPLS(spectra,
                   l = 100):

            if spectra.columns[0] != 'wave':
                print('This function needs the first column to be the axis, "wave"')
                return

            spec_list = []

            for column in tqdm(spectra.columns[1:], desc='Removing baseline from each spectrum'):
                spectrum = spectra[column]
                bl = airPLS(spectrum, lambda_ = 1)
                blr_spectrum = spectrum - bl
                blr_spectrum = blr_spectrum.tolist()
                spec_list.append(blr_spectrum)

            blr_spectra = pd.DataFrame(spec_list).T
            blr_spectra.columns = spectra.columns[1:]
            blr_spectra.insert(0, 'wave', spectra['wave'])

            return blr_spectra
        
    class Normalize():
        
        '''
        NORMALIZATION
        '''
    
        # normalize each df subset of data, then concatenate
        def normalize_regions(df_list: list,
                              method = 'l1'):

            count = 0

            # default, but leaving option open for other methods
            if method = 'l1':
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

            # limits from Anderson et al. 2021, Table 1.
            # https://doi.org/10.1016/j.sab.2021.106347

            uv = spectra[(spectra['wave'] >= 243.79) & (spectra['wave'] <= 341.36)].copy(deep=True)
            vis = spectra[(spectra['wave'] >= 379.26) & (spectra['wave'] <= 464.54)].copy(deep=True)
            vnir_1 = spectra[(spectra['wave'] >= 537.57) & (spectra['wave'] <= 619.82)].copy(deep=True)
            vnir_2 = spectra[(spectra['wave'] >= 620.08) & (spectra['wave'] <= 712.14)].copy(deep=True)
            vnir_3 = spectra[(spectra['wave'] >= 712.17) & (spectra['wave'] <= 852.77)].copy(deep=True)

            df_list = [uv, vis, vnir_1, vnir_2, vnir_3]

            normed_spectra = Normalize.normalize_regions(df_list)

            return normed_spectra

        # normalize by ChemCam method
        def norm3_CL(spectra):

            uv = spectra[(spectra['wave'] >= 246.68) & (spectra['wave'] <= 338.42)].copy(deep=True)
            vis = spectra[(spectra['wave'] >= 387.9) & (spectra['wave'] <= 469.1)].copy(deep=True)
            vnir = spectra[(spectra['wave'] >= 492.65) & (spectra['wave'] <= 849.1)].copy(deep=True)

            df_list = [uv, vis, vnir]

            normed_spectra = Normalize.normalize_regions(df_list)

            return normed_spectra
        
        # normalize by SuperLIBS 10K method
        def norm3_SL_10K(spectra):

            uv = spectra[(spectra['wave'] >= 233.12) & (spectra['wave'] <= 351.35)].copy(deep=True)
            vis = spectra[(spectra['wave'] >= 370.16) & (spectra['wave'] <= 479.07)].copy(deep=True)
            vnir = spectra[(spectra['wave'] >= 498.14) & (spectra['wave'] <= 859.44)].copy(deep=True)

            df_list = [uv, vis, vnir]

            normed_spectra = Normalize.normalize_regions(df_list)

            return normed_spectra
        
        # normalize by SuperLIBS 18K method
        def norm3_SL_18K(spectra):

            uv = spectra[(spectra['wave'] >= 233.12) & (spectra['wave'] <= 351.35)].copy(deep=True)
            vis = spectra[(spectra['wave'] >= 370.16) & (spectra['wave'] <= 479.07)].copy(deep=True)
            vnir = spectra[(spectra['wave'] >= 508.3) & (spectra['wave'] <= 869.2)].copy(deep=True)

            df_list = [uv, vis, vnir]

            normed_spectra = Normalize.normalize_regions(df_list)

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

    # convert data to dict of train/test dfs per fold
    def make_data_dict(self, var, fold_col, test_fold=None):
        
        # find minimum number of samples in the folds, 
        # to specify model params later
        n_samples_list = []

        # remove test data if using it
        if test_fold:
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

        if test_fold:
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

        else:
            train_meta = self.meta[(self.meta[fold_col] != -1) &
                              (~self.meta[fold_col].isnull())]
            y_train = train_meta[var].values
            train_names = train_meta['pkey'].values
            X_train = select_spectra(self.spectra, train_names)

            return train_names, X_train, y_train

##########################################        

class Model():
    
    '''
    Functions that optimize regression models
    '''
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
            
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
        for n_components in tqdm(component_range, desc='component value'):
            # define model
            model = PLSRegression(n_components = n_components, scale=False)
            # run CV and get RMSE
            temp_rmsecv = Model.run_CV(self, model)
            # add results to dictionary
            cv_dict[temp_rmsecv] = n_components
            
        rmsecv = min(list(cv_dict.keys()))
        component = cv_dict[rmsecv]
        model = PLSRegression(n_components = component, scale=False)
        
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from {component}-component model')
            
        return component, rmsecv, model

    def run_LASSO(self, num_alphas):
        
        alpha_range = np.logspace(-10, 1, num_alphas)

        cv_dict = dict()
        for alpha in tqdm(alpha_range, desc='alpha value'):
            model = Lasso(alpha=alpha)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = alpha

        rmsecv = min(list(cv_dict.keys()))
        alpha = cv_dict[rmsecv]
        model = Lasso(alpha=alpha)
        
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(alpha,5)}')
            
        return alpha, rmsecv, model
            
    def run_Ridge(self, num_alphas):
        
        alpha_range = np.logspace(-10, 1, num_alphas)
        
        cv_dict = dict()
        for alpha in tqdm(alpha_range, desc='alpha value'):
            model = Ridge(alpha=alpha)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = alpha
        
        rmsecv = min(list(cv_dict.keys()))
        alpha = cv_dict[rmsecv]
        model = Ridge(alpha=alpha)
        
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(alpha,5)}')
            
        return alpha, rmsecv, model
    
    def run_ElasticNet(self, num_alphas):
        
        # suggested by documentation to skew to lasso
        ratio_range = [.1, .5, .7, .9, .95, .99, 1]
        # slightly raise min because takes longer
        alpha_range = np.logspace(-7, 1, num_alphas)

        cv_dict = dict()
        for ratio in tqdm(ratio_range, desc='L1 ratio', leave=False):
            for alpha in tqdm(alpha_range, desc='alpha value', leave=False):
                model = ElasticNet(alpha=alpha, l1_ratio=ratio)
                temp_rmsecv = Model.run_CV(self, model)
                cv_dict[temp_rmsecv] = [alpha, ratio]
        
        rmsecv = min(list(cv_dict.keys()))
        params = cv_dict[rmsecv]
        model = ElasticNet(alpha=params[0], l1_ratio=params[1])
        
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an alpha of {round(params[0],5)} and an l1_ratio of {params[1]}')
        param = f'alpha={params[0]} l1_ratio={params[1]}'
            
        return param, rmsecv, model    
        
           
    def run_SVR_linear(self, num_epsilons):
        
        # smaller range here
        epsilon_range = np.logspace(-4, 1, num_epsilons)

        cv_dict = dict()
        for epsilon in tqdm(epsilon_range, desc='epsilon value'):
            model = SVR(kernel='linear', epsilon=epsilon)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = epsilon
        
        rmsecv = min(list(cv_dict.keys()))
        epsilon = cv_dict[rmsecv]
        model = SVR(kernel='linear', epsilon=epsilon)
        
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an epsilon of {round(epsilon,5)}')
            
        return epsilon, rmsecv, model
    
    def run_SVR_poly(self, num_epsilons, poly_deg):
        
        print(f'Currently using a polynomial degree of {poly_deg}')
        
        epsilon_range = np.logspace(-4, 1, num_epsilons)

        cv_dict = dict()
        for epsilon in tqdm(epsilon_range, desc='epsilon value'):
            model = SVR(kernel='poly', degree=poly_deg, epsilon=epsilon)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = epsilon
        
        rmsecv = min(list(cv_dict.keys()))
        epsilon = cv_dict[rmsecv]
        model = SVR(kernel='poly', degree=poly_deg, epsilon=epsilon)
        
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with an epsilon of {round(epsilon,5)}')
            
        return epsilon, rmsecv, model
    
    def run_PCR_linear(self):
        
        print('PCR-lin does not optimize')
        # removed component range because different thing
        model = Pipeline([('PCA', PCA()), ('linear', LinearRegression())])
        rmsecv = Model.run_CV(self, model)
        
        print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
            
        return 'NA', rmsecv, model
    
    def run_PCR_poly(self, poly_deg):
        
        print('PCR-py does not optimize')
        #print(f'Currently using a polynomial degree of {poly_deg}')

        pca = KernelPCA(kernel='poly', degree=poly_deg)
        model = Pipeline([('PCA',pca), ('linear', LinearRegression())])
        rmsecv = Model.run_CV(self, model)
        
        print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
        
        return 'NA', rmsecv, model
    
    def run_OMP(self):
        
        print('OMP does not optimize')
        model = OrthogonalMatchingPursuit()
        rmsecv = Model.run_CV(self, model)
        
        print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
        
        return 'NA', rmsecv, model
    
    def run_RF(self):
        
        feat_range = ['sqrt', 'log2'] # `None` took long

        cv_dict = dict()
        for feat in tqdm(feat_range, desc='max features'):
            model = RandomForestRegressor(max_features=feat)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = feat
        
        rmsecv = min(list(cv_dict.keys()))
        feat = cv_dict[rmsecv]
        model = RandomForestRegressor(max_features=feat)
        
        print(f'\tLowest RMSE-CV of {round(rmsecv,2)} obtained from model with {feat} max features')
            
        return feat, rmsecv, model
    
    def run_GBR(self):
        
        feat_range = ['sqrt', 'log2'] # `None` took long

        cv_dict = dict()
        for feat in tqdm(feat_range, desc='max features'):
            model = GradientBoostingRegressor(random_state=0, max_features=feat)
            temp_rmsecv = Model.run_CV(self, model)
            cv_dict[temp_rmsecv] = feat
        
        rmsecv = min(list(cv_dict.keys()))
        feat = cv_dict[rmsecv]
        model = GradientBoostingRegressor(random_state=0, max_features=feat)
        
        print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model with {feat} max features')
        
        return feat, rmsecv, model
    
    def run_OLS(self):
        
        print('OLS does not optimize')
        model = LinearRegression()
        rmsecv = Model.run_CV(self, model)
        
        print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
        
        return 'NA', rmsecv, model
    
    def run_kNN(self, max_neighbors):
        
        neighbor_range = np.arange(1,max_neighbors)
        weight_range = ['uniform','distance']

        cv_dict = dict()
        for neighbor in tqdm(neighbor_range, desc='# neighbors'):
            for weight in weight_range:
                model = KNeighborsRegressor(n_neighbors=neighbor, weights=weight)
                temp_rmsecv = Model.run_CV(self, model)
                cv_dict[temp_rmsecv] = [neighbor, weight]
        
        rmsecv = min(list(cv_dict.keys()))
        params = cv_dict[rmsecv]
        model = KNeighborsRegressor(n_neighbors=params[0], weights=params[1])
        
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
        plt.savefig(f'{path}\\{var}_{method}_coefs_plot.jpg', dpi=600)
        plt.savefig(f'{path}\\{var}_{method}_coefs_plot.eps', dpi=600)
        plt.close()
        
    # predicted vs true scatter plot with 1:1 line
    def pred_true(df, var, method, type, rmse, adj_r2, path):
        
        actual_col = f'{var}_actual'
        pred_col = f'{var}_pred'

        # PARAMETERS
        size=14 # font size
        color = 'black' #'#3f997c' # dot color
        opacity = 0.6 # dot opacity

        # get plot limits for 1:1
        plt_max = get_max(max(max(df[actual_col].values), max(df[pred_col].values)))       
        plt_min = get_min(min(min(df[actual_col].values), min(df[pred_col].values)))

            
        # get X limits
        x_min = get_min(min(df[actual_col].values))
        x_max = get_max(max(df[actual_col].values))
        
        # get \y limits
        y_min = get_min(min(df[pred_col].values))
        y_max = get_max(max(df[pred_col].values))

        # make plot
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(df[actual_col], df[pred_col], c=color, alpha=opacity)
        ax.plot([plt_min,plt_max], [plt_min,plt_max], 'k--')
        ax.plot([plt_min,plt_max], [0,0], 'k')
        plt.xlim(x_min,x_max)
        plt.ylim(y_min, y_max)

        plt.title(f'{type} RMSE: {round(rmse, 2)}    Adj. R2: {round(adj_r2, 2)}', fontsize=size)
        ax.set_ylabel(f'Predicted {var}', fontsize=size)
        ax.set_xlabel(f'Actual {var}', fontsize=size)

        # save plot
        plt.tight_layout()
        plt.savefig(f'{path}\\{var}_{method}_{type}_pred_true_plot.jpg', dpi=600)
        plt.savefig(f'{path}\\{var}_{method}_{type}_pred_true_plot.eps', dpi=600)
        plt.close()