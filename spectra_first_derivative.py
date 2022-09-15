import pandas as pd
import os
from model_tools import check_csv

'''
Code by Cai Ytsma (cai@caiconsulting.co.uk)
Last updated 15 September 2022

This code calculates the first derivative of all the spectra.
The first column should be the X axis, called 'wave'.
The subsequent columns are the spectra to be taken the derivative of.
'''

# input information
prompt = 'Path location and file name: (e.g. C:\Documents\spectra.csv) '
path = check_csv(input(prompt))
while not os.path.exists(path):
    print(f'Error: path {path} does not exist')
    path = check_csv(input(prompt))

# read in data
spectra = pd.read_csv(path)
sample_cols = [col for col in spectra.columns if col != 'wave']

# get derivative per spectrum
deriv_spectra = []
for spectrum in sample_cols:
    deriv_spectrum = list(spectra.diff().eval(f'{spectrum}/wave'))
    deriv_spectra.append(deriv_spectrum)

# convert to df
deriv_df = pd.DataFrame(deriv_spectra).T
deriv_df.columns = sample_cols
deriv_df.insert(0,'wave',spectra['wave'])
# drop first row (no data)
deriv_df = deriv_df.drop(index=0).reset_index(drop=True)

# export revised spectra
folder = '\\'.join(path.split('\\')[:-1])
filename = path.split('\\')[-1][:-4]
new_filename = f'{filename}_first_derivative.csv'
deriv_df.to_csv(f'{folder}\\{new_filename}', index=False)

print('Exported', new_filename)