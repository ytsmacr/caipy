import pandas as pd
import os
import re
from statistics import median, mean

'''
Code to compile relative uncertainty values from 
calculate_sensitivity.py

by Cai Ytsma (cai@caiconsulting.co.uk)
Last updated 28 September 2022
'''

folder = input('Folder containing subfolders of model results: ')
while os.path.exists(folder) is False:
    raise ValueError('Path does not exist')
    folder = input('Folder containing subfolders of model results: ')
    
folder_list = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder,f))]
if len(folder_list)==0:
    raise ValueError('No subfolders in this folder')
print(len(folder_list), 'folders found')

done_any = False
filename = 'average_relative_channel_uncertainty.csv'

count = 0
folder_list = []
type_list = []
mean_list = []
med_list = []

for root, dirs, files in os.walk(folder):

    for fname in files:
        if fname != filename:
            continue
        fpath = os.path.join(root,fname)
        ffolder = root.split('\\')[-1]

        df = pd.read_csv(fpath)
        
        # all
        med_all = median(df['avg_rel_stdev'])
        mean_all = mean(df['avg_rel_stdev'])

        med_list.append(med_all)
        mean_list.append(mean_all)
        folder_list.append(ffolder)
        type_list.append('relative uncertainty')

        # below threshold
        thresh_col = [c for c in df.columns if 'below' in c][0]

        df = df[df[thresh_col]=='yes']
        med_below = median(df['avg_rel_stdev'])
        mean_below = mean(df['avg_rel_stdev'])

        med_list.append(med_below)
        mean_list.append(mean_below)
        folder_list.append(ffolder)
        type_list.append('relative unceratinty '+thresh_col)
        
        # get overall sensitivity
        df = pd.read_csv(os.path.join(root,'sensitivity_results.csv'), nrows=3)
        med_sens = df['*Sensitivity Information*'][0]
        mean_sens = df['*Sensitivity Information*'][1]
        
        med_list.append(med_sens)
        mean_list.append(mean_sens)
        folder_list.append(ffolder)
        type_list.append('overall sensitivity')

        count += 1

if count > 0:
    print(f'{count} relevant sets of files found')
    
    results = pd.DataFrame({
        'folder':folder_list,
        'type':type_list,
        'median':med_list,
        'mean':mean_list
    })
    results.to_csv(f'{folder}\\cumulative_{filename}', index=False)
    
    print(f'exported cumulative_{filename}')
    done_any=True

if not done_any:
    raise ValueError(f'No suitable files in these folders to combine. ({filename})')