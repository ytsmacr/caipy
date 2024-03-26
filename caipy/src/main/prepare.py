# TODO:
# -Spectra preprocessing
# -meta(done)/spectra/both scripts
import argparse
import os
import subprocess
import sys
#class feed args to global.py
from globals import *
from utilities import *
from stratify_samples import stratification
import parallel as parallel
#procedure preparation class (process arguments, clean data if necessary)
class UserArgs():
    def askArgs(self, sysargs=None):
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
        args, unknown = parser.parse_known_args(sysargs)
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

        #set spectra, meta data frames, from directories. If arguments below not specified, they are asked for
        dataLoadingUtilities.directories(data_folder, outpath, spectra_path, meta_path)
        # show the progress bars and results of CV
        if hide_progress is not None:
            Globals.hide_progress = hide_progress
        if max_components_ is not None:
            Globals.max_components_ = max_components_ 
        if num_params_ is not None:
            Globals.num_params_ = num_params_
        if poly_deg_ is not None:
            Globals.poly_deg_ = poly_deg_
        if max_neighbors_ is not None:
            Globals.max_neighbors_ = max_neighbors_
           
    #check meta/spectra in same order, adds axis wavelengths to globals
    def meta_spectra_align(self, meta, spectra):
        #cleaningUtilities.meta_spectra_align(meta, spectra)
        check = list(spectra.columns[1:]) == list(meta['pkey'].values)
        if not check:    
            raise ValueError('Spectra and metadata samples need to be in same order')
        Globals.axis = list(Globals.spectra['wave'].values)

#main method that can handle if BOTH spectra and metadata supplied, or either or
def main(args = sys.argv):
    run = UserArgs()
    run.askArgs()
    #meta only
    if Globals.spectra is None and Globals.meta is not None:
        metaUtilities.check_meta_format(Globals.meta, Globals.meta_path)  
        metaUtilities.clean_meta_for_negatives(Globals.meta, Globals.var_to_run)
        metaUtilities.generate_meta_statistics()
    #spectra only
    elif Globals.meta is None and Globals.spectra is not None:
        preprocessingUtilities.spectra_options(Globals.spectra)
        preprocessingUtilities.apply_spectra_args()
    #have both:
    elif Globals.meta is not None and Globals.spectra is not None:
        #must check they align before any further changes to either 
        run.meta_spectra_align(Globals.meta, Globals.spectra)
        #meta processing
        metaUtilities.check_meta_format(Globals.meta, Globals.meta_path)
        #spectra processing
        preprocessingUtilities.spectra_options(Globals.spectra)
        preprocessingUtilities.apply_spectra_args(Globals.spectra)
    else:
        print("wrong function idiot")

if __name__ == '__main__':  
    main()

#is spectra recived as none, but meta exists? clean it, stratify if necessary
def meta_only(args = sys.argv):
    run = UserArgs()
    run.askArgs()
    metaUtilities.check_meta_format(Globals.meta, Globals.meta_path)  
    metaUtilities.clean_meta_for_negatives(Globals.meta, Globals.var_to_run)
    metaUtilities.generate_meta_statistics()
    if Globals.spectra is not None:
        print("spectra data recieved, but not meddled with since you ran a metadata only script!")

#only spectra file supplied, to which preprocessing functions executed
#spectra-exclusive optional arguments:
        # [--resample int] should it be resazmpled somehow? if so number from 2 value dict for resample
        # [--blr boolean] baseline removal?
        # [--normalization int] number for which normalization method
def spectra_only(args = sys.argv):
    run = UserArgs()
    run.askArgs()
    preprocessingUtilities.spectra_options(Globals.spectra)
    preprocessingUtilities.apply_spectra_args(Globals.spectra)
    if Globals.meta is not None:
        print("meta data recieved, but not meddled with since you ran a spectra only script!")

#python3 script_sequence.py  -f /Users/sanjanayasna/wintern/caipy/test_data -o /Users/sanjanayasna/wintern/caipy/test_results -s one_spectra.csv 