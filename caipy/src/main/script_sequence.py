import argparse
import sys
import prepare as prepare
import models as models
import analyze as analyze
import apply_model
from globals import *
import os
import subprocess
import prepare
import stratify_samples
# RUN MULTIPLE SCRIPTS CONSECUTIVELY
# example input...
# poetry run preprocess build_models -f C:\Users\arifu\Winternship\caipy\test_data -o C:\Users\arifu\Winternship\caipy\test_results -s spectra_multi.csv -m meta_multi.csv -std True 


#main method first has a parser that contains ALL possible options between parsers of different scripts (anything you can possibly supply in command line for any script)
def main(args = sys.argv):
    #dict of valid script names (based on poetry script names/file names)
    operations_dict = {"preprocess": prepare.main, 
                       "prepare_spectra_only" : prepare.spectra_only, 
                      "prepare_meta_only": prepare.meta_only, 
                      "just_stratify": stratify_samples.main, 
                      "models": models.main, 
                      "apply_models": apply_model.main,
                      "analyze": analyze.main
                      }
    
    #get operations list of scripts to execute
    for arg in args[1:]:
        if arg in operations_dict:
            print("Executing", arg, "script...")
            operations_dict[arg]()
        #if arg isn't a script, what follows pertains to script execution, break for you have gone through all script names
        else: break

if  __name__ == '__main__':  
    main()