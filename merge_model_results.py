import pandas as pd
import os
import re
import argparse

'''
Code to compile model result files from 
spectral_regression_modelling.py

by Cai Ytsma (cai@caiconsulting.co.uk)
Last updated 30 November 2023
'''

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, default=None, help='Path of folder containing subfolders of model results')
args=parser.parse_args()
folder = args.folder.replace("'","")

if folder is None:
    folder = input('Folder containing subfolders of model results: ')
while os.path.exists(folder) is False:
    raise ValueError('Path does not exist')
    folder = input('Folder containing subfolders of model results: ')
    
folder_list = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder,f))]
if len(folder_list)==0:
    raise ValueError('No subfolders in this folder')
print(len(folder_list), 'folders found')

done_any = False
filename = 'modelling_results.csv'
df_list = []
for root, dirs, files in os.walk(folder):
    for fname in files:
        if fname != filename:
            continue
        fpath = os.path.join(root,fname)
        ffolder = root.split('\\')[-1]

        df = pd.read_csv(fpath)
        df.insert(loc=0, column='folder', value=ffolder)
        df_list.append(df)
        
if len(df_list) > 0:
    print(f'{len(df_list)} {filename} files found')
    merged_data = pd.concat(df_list)
    merged_data.to_csv(f'{folder}\\cumulative_{filename}', index=False)
    print(f'exported cumulative_{filename}')
    done_any=True
    
if not done_any:
    raise ValueError('No suitable files in these folders to merge. (modelling_results.csv)')
        