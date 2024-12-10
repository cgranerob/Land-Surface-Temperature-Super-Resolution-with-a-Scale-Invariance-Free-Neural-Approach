#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Script to be run after process_modis.py. Its purpose is not to delete files 
downloaded, it will leave the data intact. 

It will work directly on the pairs_*.csv files. to create the train and test datasets.

Else
----
@author: Romuald Ait Bachir
"""
import utils as us
import pandas as pd
import numpy as np

from random import choices, seed
import os
from json import dump as json_dump

# In this part, we prepare the work that will be done in dataset.py
# The .csv file (maybe filtered) are transformed into two new files
random_seed = 42

csv_day_df = pd.read_csv("./data/pairs_day.csv")
csv_day_df.drop(columns=csv_day_df.columns[0], axis=1, inplace=True)
csv_day_df['time'] = csv_day_df['LST'].apply(lambda x : 'D')

# We assign now another column which will dictate if a value is in train, test or validation
seed(random_seed)

val = ['Train', 'Val']
# proportions = [0.9, 0.1] 
proportions = [0.6, 0.4]

# This is done now in order to have the exact data used to train the model A and the model B.
csv_day_df['split'] = csv_day_df['LST'].apply(lambda x : choices(val, proportions)[0])

# Finally, the two last .csv files are made. These two .csv files will be each, 
# respectively used by ModisDatasetA and ModisDatasetB.
# ModisDatasetA
moddsA = csv_day_df
moddsA = moddsA.sample(frac=1, random_state = random_seed).reset_index(drop=True)

# For this dataset, the two image columns will be concatenated.
moddsA_ndvi = moddsA.copy()
moddsA_ndvi.drop(columns=moddsA_ndvi.columns[0], axis=1, inplace=True) # Dropping the LST
moddsA_ndvi['time'] = moddsA_ndvi['time'].apply(lambda x : 'NDVI')
moddsA_ndvi = moddsA_ndvi.rename(columns = {"NDVI": "IMG"})

moddsA.drop(columns=moddsA.columns[1], axis=1, inplace=True) # Dropping the NDVI
moddsA = moddsA.rename(columns = {"LST": "IMG"})

moddsA = pd.concat([moddsA, moddsA_ndvi])
moddsA = moddsA.sample(frac=1, random_state = random_seed).reset_index(drop=True)
moddsA = moddsA.drop_duplicates(subset = ['IMG']) # In case there are duplicates.
moddsA.to_csv(os.path.join(os.getcwd(),"data/ModisDatasetA.csv"))

# ModisDatasetB -
moddsB = csv_day_df
moddsB = moddsB.sample(frac=1, random_state = random_seed).reset_index(drop=True)
moddsB.to_csv(os.path.join(os.getcwd(),"data/ModisDatasetB.csv"))

# The .csv files are now ready. No we need to remake the old pytorch datasets.

#%% This should run in 1 minute max
# Now we can get the statistics of the train dataset in order to do the normalization.

df_path = './data/ModisDatasetB.csv'
save_path = './data/'

df = pd.read_csv(df_path)
df.drop(columns=df.columns[0], axis=1, inplace=True)
df = df.loc[df['split'] == 'Train'] # Filtering to get only the train split

df = df.loc[df['time'] == 'D']

# Loading all the images
df['LST'] = df['LST'].apply(lambda x : us.read_GeoTiff(x)[0])

liste = list(df['LST'])

stats = {}
stats['maxi'] = max([np.max(i) for i in liste])
stats['mini'] = min([np.min(i) for i in liste])

a = np.zeros((64,64*len(liste)))
for i, mat in enumerate(liste):
    a[:,i*64:(i+1)*64] = mat
stats['mean_lst'] = np.mean(a)
stats['std_lst'] = np.std(a)

df['NDVI'] = df['NDVI'].apply(lambda x : us.read_GeoTiff(x)[0])

liste = list(df['NDVI'])
a = np.zeros((256,256*len(liste)))
for i, mat in enumerate(liste):
    a[:,i*256:(i+1)*256] = mat
stats['mean_ndvi'] = np.mean(a)
stats['std_ndvi'] = np.std(a)

with open(os.path.join(save_path, "statistics.json"), 'w') as outfile:
    json_dump(eval(str(stats)), outfile)

# The maximum maxi is in either the validation or the test data!
#  Wondering if the train maxi being lower than the val/test maxi will have
# a real impact on the performances.
