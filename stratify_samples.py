import pandas as pd
import numpy as np
import os

from model_tools import check_csv

'''
Code by Cai Ytsma (ytsmacr@gmail.com)
Last updated 27/07/2022

This code takes stratifies samples into a set number of folds, 
grouping by sample name and sorting by metadata value

Input file format:
'pkey', 'Sample_Name' (to group by), {variable1}, {variable2}, etc.

Output file format:
'pkey', 'Sample_Name', {variable1}, '{variable1}_Folds', etc.
'''

# input information
meta_file = check_csv(input('Path location and file name: (e.g. C:\Documents\meta_file.csv) '))
while not os.path.exists(meta_file):
    print(f'Error: path {meta_file} does not exist')
    meta_file = check_csv(input('Path location and file name: (e.g. C:\Documents\meta_file.csv) '))
    
fold_input = input('How many folds do you want? (Press ENTER for default of 5) ')
n_folds = 5 if fold_input == "" else int(fold_input)

meta = pd.read_csv(meta_file).reset_index()
for c in ['pkey', 'Sample_Name']:
    if c not in meta.columns:
        raise ValueError(f"Error: column '{c}' must exist")

var_list = [c for c in meta.columns if c not in ['index', 'pkey', 'Sample_Name']]

for var in var_list:

    meta = meta.sort_values([var,'Sample_Name'], ignore_index=True)

    fold_list = []
    for i in meta.index:

        # set fold to -1 (aka to remove) if no value
        if pd.isna(meta.loc[i, var]):
            fold_list.append(-1)

        elif i == 0:
            fold_list.append(1)

        else:
            current_sample = meta.loc[i,'Sample_Name']
            last_sample = meta.loc[i-1,'Sample_Name']
            last_fold = fold_list[-1]
            
# probably dont need this since the nans will always be at the end
#            idx = 2
#            while last_fold == -1:
#                last_fold = fold_list[-idx]
#                idx+=1
#                if idx > len(fold_list):
#                    last_fold = n_folds #(i.e. will force to be 1 later)

            if current_sample == last_sample:
                fold_list.append(last_fold)
            else:
                current_fold = int(last_fold + 1 if last_fold < n_folds else 1)
                fold_list.append(current_fold)

    meta[f'{var}_Folds'] = fold_list
    
meta = meta.sort_values('index').reset_index(drop=True)
meta.drop(columns='index', inplace=True)

folder = '\\'.join(meta_file.split('\\')[:-1])
filename = meta_file.split('\\')[-1][:-4]
new_filename = f'{filename}_stratified.csv'
meta.to_csv(f'{folder}\\{new_filename}', index=False)

print('Exported', new_filename)
