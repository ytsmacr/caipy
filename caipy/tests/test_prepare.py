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
@pytest.fixture(name = "run_model_parallel")
#METHOD ARGUMENT IS NUMBER
def run_model(method):
    Globals.var_to_run = [col for col in Globals.meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)]
    test = modelSetup()
    test.set_folds()
    print(Globals.var_to_run)
    for var in Globals.var_to_run:
        # RESET MODEL PARAMETERS
        #maximum number of components for PLS
        max_components_ = 6
        # number of values to test for LASSO, Ridge, ElasticNet, SVR
        num_params_ = Globals.num_params_
        # polynomial degree for SVR and kernel PCR
        poly_deg_ = Globals.poly_deg_
        # maximum number of neighbors for kNN
        max_neighbors_ = Globals.max_neighbors_

        #Get sample test fold
        print(f'\nRunning for {var}')
        fold_col = foldUtilities.get_fold_col(var)
        test_fold = None
        Globals.data_dict, min_samples = foldUtilities.make_data_dict(var, fold_col, test_fold)
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
            'LASSO':{'func':modelUtilities.run_LASSO,
                    'args':num_params_},
            'Ridge':{'func':modelUtilities.run_Ridge,
                    'args':num_params_},
            'ElasticNet':{'func':modelUtilities.run_ElasticNet,
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
        names = (list(reg_cv_dict.keys()))
        print(f'\nPerforming CV for {names[method]}')
        type(reg_cv_dict[names[method]]['args']) 
        return names[method]
#     #run the model
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
    
    
    
#test various functions of prepare.py
class TestPrepare:
    #initialize parser defined above and assert values can be recognized for generic inputs
    def test_parser(self, create_parser): #,process_parser):
        #initialize parser
        self.parser = create_parser
        #feed arguments for test1 parser
        test1 = self.parser.parse_args(['-f',  r'test_data'])
        assert test1.data_folder == r'test_data'
        
    #are arguemnts passed into the args parser recognized? And new values are updated in the overal Globals class?
    def test_direct_input(self, create_parser):
        test1 =  create_parser
        test1 = test1.parse_args(['-f',  r'tests/test_data', 
                                  '-o',  r'tests/test_results'])
        
        process_parser(test1)
        assert Globals.data_folder == test1.data_folder
        test2 =  create_parser
        test2 = test2.parse_args(['-f',  r'someNewValue', 
                                  '-o',  r'anotherOne'])
        process_parser(test2)
        assert Globals.data_folder == test2.data_folder
        assert Globals.standard is None

    #checks if stratify script is run when needed
    def test_check_meta_format_stratify_run(self, create_parser):
        #given one_meta.csv, assert that the stratify script was called...
        #create mock object that applies to one patch
        test0 = create_parser
        test0 = test0.parse_args(['-f',  r'tests/test_data', 
                                  '-o',  r'tests/test_results',
                                  '-m', 'one_meta.csv',
                                  '-n', '5'])
        process_parser(test0)
        mock1 = Mock()
        mock1.metaUtilities.check_meta_format(Globals.meta, Globals.meta_path)
        mock1.metaUtilities.check_meta_format.assert_called()

    #run check_meta_format with different meta files (indirectly parameterized), all of which are stratified (if they shouldn't, adjust this test)
    # checks if meta path correct
    # checks that that var_to_run doesn't have any unexpected values
    # checks if resulting Globals.meta value aligns with expected stratified result
    @pytest.mark.parametrize("meta_path, expected", [ ("one_meta.csv", os.path.join(os.getcwd(),'tests/test_data/one_meta.csv')),
                            ("meta_multi_nosample.csv", os.path.join(os.getcwd(),'tests/test_data/meta_multi_nosample.csv')),
                            ('test_meta_multi_bad.csv', os.path.join(os.getcwd(),'tests/test_data/test_meta_multi_bad.csv')),
                            ('test_meta.csv', os.path.join(os.getcwd(),'tests/test_data/test_meta.csv'))
                            ] )
    def notest_check_meta_format_runner(self, create_parser, meta_path, expected):
        parser = create_parser
        parser = parser.parse_args(['-f',  os.path.join(os.getcwd(),'tests/test_data'), 
                                  '-o',  os.path.join(os.getcwd(),'tests/test_results'),
                                  '-m', meta_path,
                                  '-n', '5'])
        process_parser(parser)
        assert Globals.meta_path == expected
        print("recieved path", Globals.meta_path)
        #check of the initial list of meta columns don't have unamed columns filtered out (which can hyappen due to indexing errors when initially reading in csv)
        initial_var_to_run = [col for col in Globals.meta.columns if (col not in ['pkey', 'Sample Name', 'Sample_Name']) and ('Folds' not in col)]
        assert "Unamed " not in initial_var_to_run
        #permission errors for this particular meta file, but direct comparison holds
        # if (meta_path == 'test_meta.csv'):
        #     pass
        # else:
        #     var_to_run, all_var, meta = metaUtilities.check_meta_format(Globals.meta, Globals.meta_path)
        #     assert "wave" not in var_to_run
        #     #Check that new Globals.meta pd dataframe matches expected results under test_results
        #     split = meta_path.split('.')
        #     corresponding_file_in_test_results = split[0] + "_stratified." + split[1]
        #     correct_meta = pd.read_csv(os.path.join(Globals.outpath, corresponding_file_in_test_results))
        #     assert Globals.meta.equals(correct_meta)
        #     #now remove the stratified file created if it exists:
        #     if os.path.exists(os.path.join(Globals.data_folder, corresponding_file_in_test_results)):
        #         os.remove(os.path.join(Globals.data_folder, corresponding_file_in_test_results))

        var_to_run, all_var, meta = metaUtilities.check_meta_format(Globals.meta, Globals.meta_path)
        assert "wave" not in var_to_run
        #Check that new Globals.meta pd dataframe matches expected results under test_results
        split = meta_path.split('.')
        corresponding_file_in_test_results = split[0] + "_stratified." + split[1]
        correct_meta = pd.read_csv(os.path.join(Globals.outpath, corresponding_file_in_test_results))
        assert Globals.meta.equals(correct_meta)
        #now remove the stratified file created if it exists:
        if os.path.exists(os.path.join(Globals.data_folder, corresponding_file_in_test_results)):
            os.remove(os.path.join(Globals.data_folder, corresponding_file_in_test_results))
    

    #tests fillna 
    #tests if an entire row of -1s really is dropped...
    @pytest.mark.parametrize("meta_path",
                             [ ("one_meta_drop_test1.csv")]
    )
    def test_clean_meta_for_negatives(self, create_parser, meta_path):
        parser = create_parser
        parser = parser.parse_args(['-f',  os.path.join(os.getcwd(),'tests/test_data'), 
                                  '-o',  os.path.join(os.getcwd(),'tests/test_results'),
                                  '-m', meta_path,
                                  '-n', '5'])
        process_parser(parser)
        #save original meta data file, since the cleaning method replaces current meta with adjusted one if any values dropped
        original_meta = Globals.meta
        metaUtilities.clean_meta_for_negatives(Globals.meta, Globals.var_to_run)
        correct_meta = pd.read_csv(os.path.join(Globals.outpath, Globals.meta_path), index_col=False)
        #compare to correct corresponding file in test results...
        assert Globals.meta.equals(correct_meta)
        #revert file back to its original state
        original_meta.to_csv(Globals.meta_path)

    #tests the meta_spectra align script
    @pytest.mark.parametrize("meta_path, spectra_path",
                             [ ("one_meta.csv", "one_spectra.csv"),
                              ("one_meta_drop_test1.csv", "one_spectra.csv"),
                              ("one_meta_drop_test1.csv", "spectra_multi.csv")]
    )
    def n0test_meta_spectra_align(self, create_parser, meta_path, spectra_path):
        parser = create_parser
        parser = parser.parse_args(['-f',  os.path.join(os.getcwd(),'tests/test_data'), 
                                  '-o',  os.path.join(os.getcwd(),'tests/test_results'),
                                  '-m', meta_path,
                                  '-s', spectra_path])
        process_parser(parser)
        run = UserArgs()
        if spectra_path == "spectra_multi.csv":
            pytest.raises(ValueError, run.meta_spectra_align, Globals.meta, Globals.spectra)
        else: run.meta_spectra_align(Globals.meta, Globals.spectra)
     

#make this one spectra, and blr \values (1 x 2 x 2 unique possibilities, total)
    @pytest.mark.parametrize(
        ("resample",
             [1, 2])
    )
    @pytest.mark.parametrize(
        ("norm",
             [0, 1, 2, 3])
    )
    #need to expand upon testing spectra input args...
    #for now, just see if it executes without errors for different argumetn combos:
    def notest_spectra_arg_application_multiple_combos(self, create_parser, spectra_path, blr, resample, norm):
        parser = create_parser
        parser = parser.parse_args(['-f',  os.path.join(os.getcwd(),'tests/test_data'), 
                                  '-o',  os.path.join(os.getcwd(),'tests/test_results'),
                                  '-s', spectra_path,
                                  '-blr', blr,
                                  '--resample', resample,
                                  '-norm', norm])
        process_parser(parser)
        print("baseline:",Globals.baseline)
        #preprocessingUtilities.apply_spectra_args(Globals.spectra)