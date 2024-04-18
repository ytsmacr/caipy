from unittest import mock
import sys
sys.path.insert(0, 'src/main')
from prepare import *
from globals import *
from utilities import *
from stratify_samples import *
from models import *
import pytest 
import argparse
import subprocess
from unittest.mock import Mock, patch
import glob

# TEST PARALLEL.PY

@pytest.fixture(scope="class", autouse=True)
def parallel_file_search(request):
    train_pred_true_filesList = []
    for var in Globals.methods_dict.keys():
        for method in Globals.methods_dict[var]:
            methods = ['OLS', 'OMP','LASSO','Ridge', 'ElasticNet', 'PLS', 'PCR-lin', 'PCR-py', 'SVR-lin', 'SVR-py', 'RF', 'GBR', 'kNN']
            method = methods[method]
            train_pred_true_filesList.append(glob.glob(f'{request.out1}*{var}_{method}_train_pred_true.csv', recursive=True))
    return train_pred_true_filesList
@pytest.fixture(scope="class", autouse=True)
def create_parser():
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_folder', type=str, default=None, help='Path of folder with data')
    parser.add_argument('-o', '--outpath', type=str, default=None, help='Path of folder to output results')
    parser.add_argument('-s', '--spectra_path', type=str, default=None, help='Spectra filename')
    parser.add_argument('-m', '--meta_path', type=str, default=None, help='Metadata filename')
    parser.add_argument('-std', '--standard', type=bool, default=None, help='Follow a standard procedure for each variable (bool)')
    parser.add_argument('-dt', '--do_test', type=bool, default=None, help='Hold a fold out as test data (bool)')
    parser.add_argument('-mt', '--method', type=str, default=None, help=f'Number corresponding to method selection from: {method_prompt}')
    parser.add_argument('-hp', '--hide_progress', type=bool, default=None, help='Hides progress bars')
    parser.add_argument('-mc', '--max_components', type=int, default=None, help='Sets the maximum PLS components')
    parser.add_argument('-np', '--num_params', type=int, default=None, help='Sets the number of values to test for LASSO, Ridge, ElasticNet, SVR')
    parser.add_argument('-pd', '--poly_deg', type=int, default=None, help='Sets the polynomial degree for SVR and kernel PCR')
    parser.add_argument('-mn', '--max_neighbors', type=int, default=None, help='Sets the maximum number of neighbors for kNN')
    parser.add_argument('-n','--n_folds', type=int, default=None, help='num of folds in data')
    #spectra options:
    parser.add_argument('-blr', '--baseline_removal', type=bool, default=None, help='Use baseline removal on spectra data?')
    parser.add_argument('--resample', type= int, default=None, help='Resample data to match [1] or to min step [2]')
    parser.add_argument('-norm', '--normalization', type=int, default=None, help=f'Normalize spectra data using method selection from: {normalization_prompt}')
    
    return parser
#custom args processor without user interaction 
def process_parser(request):
    #weird issue that I'll likely not bother resolving anyway:
    #The below code doesn't work.

    # if getattr(request, var) is not None:
    #         Globals.var = getattr(request,var)

    # It's a weird bug that I haven't come up with a solution to yet as to why it's like that.set
    # Type definitions are strings and print out in the process, but unfortunately setting values 
    # through a generic loop doesn't do the trick
    # Original parser code that didn't work:
    # for var in vars(request):
    #     #check for what's not none
    #     if getattr(request, var) is not None:
    #         Globals.var = getattr(request,var)
    if request.data_folder is not None:
        Globals.data_folder = request.data_folder
    if request.outpath is not None:
        Globals.outpath = request.outpath
    if request.spectra_path is not None:
        spectra_path = os.path.join(request.data_folder, request.spectra_path)
        Globals.spectra_path = spectra_path
        Globals.spectra = pd.read_csv(Globals.spectra_path)
    if request.meta_path is not None:
        meta_path = os.path.join(request.data_folder, request.meta_path)
        Globals.meta_path = meta_path
        meta = pd.read_csv(Globals.meta_path)
        meta = meta.loc[:, ~meta.columns.str.contains('^Unnamed')]
        Globals.meta = meta
        Globals.var_to_run = [col for col in Globals.meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)]
    if request.n_folds  is not None:
        Globals.n_folds   = int(request.n_folds)
    if request.resample  is not None:
        Globals.resample   = int(request.resample)
    if request.normalization is not None:
        Globals.normalization   = int(request.normalization)
    if request.baseline_removal is not None:
            Globals.baseline_removal = request.baseline_removal
    
    
class TestParallel:
    #tests PLS parallel's accuracy, no test folds, simple modelling
    @pytest.mark.parametrize("meta_path, spectra_path, method, out1, out2",
                            [ ("test_meta_stratified.csv", "test_spectra.csv", 6, "tests/temp", "tests/temp1"),
                             ("meta_multi_stratified.csv", "spectra_multi.csv", 6,"tests/temp", "tests/temp1")] )
#test_meta_stratified, test_spectra
    def test_PLS_accuracy(self,create_parser, meta_path, spectra_path,method ,  out1, out2):
        parser = create_parser
        parser = parser.parse_args(['-f',  os.path.join(os.getcwd(),'tests/test_data'), 
                                  '-o',  os.path.join(os.getcwd(),out1),
                                  '-m', meta_path,
                                  '-s', spectra_path])
        process_parser(parser)
        test = modelSetup()
        test.set_folds()
        for var in Globals.var_to_run:
            Globals.methods_dict[var] = [method]
            methods_torun = list(Globals.methods_dict[var])
        Globals.max_components_ = 2
        testContinued = trainModels()
        testContinued.apply_models()
        print(Globals.methods_dict)

        #___________________
        #Calls non-parallel version of things...
        parser = create_parser
        parser = parser.parse_args(['-f',  os.path.join(os.getcwd(),'tests/test_data'), 
                                  '-o',  os.path.join(os.getcwd(),out2),
                                  '-m', meta_path,
                                  '-s', spectra_path])
        process_parser(parser)
        original = modelSetup()
        original.set_folds()
        originalContinued = trainModels()
        originalContinued.apply_models_non_parallel()
        #Check all model prediction results files and remove them, as needed
        for var in Globals.methods_dict.keys():
            for method in Globals.methods_dict[var]:
                methods = ['OLS', 'OMP','LASSO','Ridge', 'ElasticNet', 'PLS', 'PCR-lin', 'PCR-py', 'SVR-lin', 'SVR-py', 'RF', 'GBR', 'kNN']
                method = methods[method]
                parallel_file = glob.glob(f'{out1}/*{var}_{method}_train_pred_true.csv')[0]
                parallel_output = pd.read_csv(os.path.join(os.getcwd(),parallel_file), index_col = False)
                correct_file =  glob.glob(f'{out2}/*{var}_{method}_train_pred_true.csv')[0]
                correct_output =  pd.read_csv(os.path.join(os.getcwd(),correct_file), index_col = False)
                assert parallel_output.equals(correct_output)

    #tests PLS parallel's accuracy, no test folds, simple modelling
    @pytest.mark.parametrize("meta_path, spectra_path, method, out1, out2",
                            [ ("test_meta_stratified.csv", "test_spectra.csv", 5, "tests/temp", "tests/temp1"),
                             ("meta_multi_stratified.csv", "spectra_multi.csv", 5,"tests/temp", "tests/temp1")] )
#test_meta_stratified, test_spectra
    def test_ELasticNet_accuracy(self,create_parser, meta_path, spectra_path,method ,  out1, out2):
        parser = create_parser
        parser = parser.parse_args(['-f',  os.path.join(os.getcwd(),'tests/test_data'), 
                                  '-o',  os.path.join(os.getcwd(),out1),
                                  '-m', meta_path,
                                  '-s', spectra_path])
        process_parser(parser)
        test = modelSetup()
        test.set_folds()
        for var in Globals.var_to_run:
            Globals.methods_dict[var] = [method]
            methods_torun = list(Globals.methods_dict[var])
        Globals.max_components_ = 2
        testContinued = trainModels()
        testContinued.apply_models()
        print(Globals.methods_dict)

        #___________________
        #Calls non-parallel version of things...
        parser = create_parser
        parser = parser.parse_args(['-f',  os.path.join(os.getcwd(),'tests/test_data'), 
                                  '-o',  os.path.join(os.getcwd(),out2),
                                  '-m', meta_path,
                                  '-s', spectra_path])
        process_parser(parser)
        original = modelSetup()
        original.set_folds()
        originalContinued = trainModels()
        originalContinued.apply_models_non_parallel()
        #Check all model prediction results files and remove them, as needed
        for var in Globals.methods_dict.keys():
            for method in Globals.methods_dict[var]:
                methods = ['OLS', 'OMP','LASSO','Ridge', 'ElasticNet', 'PLS', 'PCR-lin', 'PCR-py', 'SVR-lin', 'SVR-py', 'RF', 'GBR', 'kNN']
                method = methods[method]
                parallel_file = glob.glob(f'{out1}/*{var}_{method}_train_pred_true.csv')[0]
                parallel_output = pd.read_csv(os.path.join(os.getcwd(),parallel_file), index_col = False)
                correct_file =  glob.glob(f'{out2}/*{var}_{method}_train_pred_true.csv')[0]
                correct_output =  pd.read_csv(os.path.join(os.getcwd(),correct_file), index_col = False)
                assert parallel_output.equals(correct_output)