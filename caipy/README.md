# Instructions for use:

cd into the caipy directory under caipy2.0:

```cd caipy2.0/caipy```

install/resolve dependencies from the pyproject.toml file:

```poetry install```

run testing scripts (recommended)

```poetry run pytest```

run a script of your choice listed in [tool.poetry.scripts], line 23 of pyproject.toml, with: 

```poetry run [script-name]```

# List of current scripts and functions:

## NOTE:
issue to resolve: for windows users, I need to change a few toml settings to get direct poetry scripts to work. Otherwise, compatable with mac and linux.
If you run from poetry, these scripts don't take in command line args properly (for some reason, backticks are removed from input paths, which isn't the case with just running from terminal. instructions for running from terminal supplied below). If using poetry run, you must supply prompted arguments one at a time. Looking to see if there is a possible workaround. 

```poetry run preprocess```

Calls *prepare.py* to take in meta and spectra data inputs supplied through terminal and clean/check data format <br />
askArgs(self): calls argparser to take overall project inputs from user <br />
check_meta_format(self, meta, meta_path): stratifies metadata file if necessary, clears out file for any negatives/nan/empty values <br />
meta_spectra_align(self, meta, spectra): check if meta/spectra files align in sample order and saves overall spectra file wavelegnth axis <br />


```poetry run just_stratify```

Calls *stratify_samples.py* to stratify the metadata file based upon how many folds user would like

```poetry run build_models```

calls *models.py*
Giant class that builds the models 
handle_standard(self) : checks/gets standard variable, and asks for user selection for which methods user would like for each model (or all models if standard), to then save in a methods dictionary <br />
set_folds(self) : checks/gets do_test variable, and if do_test, it runs a function to get the most representattive test fold per sample (assuming more than one unique fold in a sample) <br />
apply_models(self) : applies cross validation to make a data dictionary of the parts of the metadata excluding the test fold that corresponds to each run of the cross validation for a given sample. Afte getting the train/test split dictionary, it runs the respective methods from the methods dictinary generated from handle_standard() <br />

## Run from terminal:

For now, if you're using windows or would prefer to supply arguments off the bat as opposed to through function prompts.
*prepare.py* :

```python .\prepare.py -f C:\Users\arifu\Winternship\caipy\test_data ...[other arguments]```

Important arguments that need to be supplied/entered through prompting since they don't have defaults are bolded 

### prepare.py arguemnt inputs:

**[-f DATAFOLDER]** <br />
**[-o OUTPATH]**<br />
**[-s SPECTRA_NAME]** <br />
**[-m META_NAME]** <br />
[-std STANDARD] <br />
[-dt DO_TEST] <br />
[-mc MAX_COMPONENTS] <br /> 
[-np NUM_PARAMS] <br />
[-pd POLY_DEG] <br />
[-mn MAX_NEIGHBORS] <br />

### stratify.py argument inputs:


**[-m FULL_METADATA_PATH]** <br />
**[-n NUM_FOLDS] : default 5 <br />

### models.py:

**[-f DATAFOLDER]** <br />
**[-o OUTPATH]** <br />
**[-s SPECTRA_NAME]** <br />
**[-m META_NAME]**<br />
**[-std STANDARD]**<br />
**[-dt DO_TEST]**<br />
Overall methods selection will be prompted one by one







