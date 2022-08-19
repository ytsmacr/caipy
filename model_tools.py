import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from statistics import mean
from tqdm import tqdm
import matplotlib.pyplot as plt 
plt.set_loglevel('error')

from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, OrthogonalMatchingPursuit
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

'''
by Cai Ytsma (cai@caiconsulting.co.uk)
Last updated 19 August 2022
'''

# STANDALONE FUNCTIONS #

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
    
    print('Warning: convert_spectra assumes the first column is the wavelength axis and ignores it')
    
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

            # held-out data
            test_meta = temp_meta[temp_meta[fold_col] == fold].reset_index(drop=True)
            X_test = select_spectra(self.spectra, test_meta.pkey)
            y_test = test_meta[var].values

            # add datasets to dictionary
            data_dict[fold] = {'train_spectra':X_train,
                               'train_metadata':y_train,
                               'test_spectra':X_test,
                               'test_metadata':y_test}

        return data_dict

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
    
    def run_PCR(self):
        
        print('PCR does not optimize')
        # removed component range because different thing
        model = Pipeline([('PCA', PCA()), ('linear', LinearRegression())])
        rmsecv = Model.run_CV(self, model)
        
        print(f'\tRMSE-CV of {round(rmsecv,2)} obtained from model')
            
        return 'NA', rmsecv, model
    
    def run_kernel_PCR(self, poly_deg):
        
        print('K-PCR does not optimize')
        print(f'Currently using a polynomial degree of {poly_deg}')

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
        
        actual_col = [col for col in df.columns if 'actual' in col][0]
        pred_col = [col for col in df.columns if 'pred' in col][0]

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