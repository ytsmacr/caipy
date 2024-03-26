
import pandas as pd
#Class variables, shared across instances (each individual executable script sohuld feed into class variables under globals standalone class)
class Globals():
    data_folder = None
    outpath = None
    spectra_path = None
    meta_path = None
    all_files = None
    spectra = pd.DataFrame()
    meta = pd.DataFrame()
    var_to_run = []
    hide_progress = False 
    max_components_ = 30
    num_params_ = 30
    poly_deg_ = 2
    max_neighbors_ = 40
    #...........
    axis = []
    methods_dict = dict()
    standard = None
    do_test = None
    do_plot_test = False #get_most_representative_folds, if you plot these. TO DO: INCORPORATE
    test_fold_per_var = dict()
    non_test_folds_per_var = dict()
    data_dict = dict()
    temp_meta = dict() #for each variable, all data excluding test_fold selelcted rows
    n_folds = None
    #spectra-only prepare.py
    baseline = None
    resample = None
    normalization = None
    #loq scirpt from analyze.py and models.py
    do_loq = None
    sensitivity = None
    #solely for parallel processing aid
    num_alphas = None



