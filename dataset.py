#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
File containing the definition of the datasets inheriting from the library
torch.utils.data.Dataset. They are used during the training of the models. 

This script contains: 
    - ModisDatasetA, dataset initially defined but not used in the end. 
    - ModisDatasetB, dataset used to train the Super Resolution model.

# Note: The argument time in the constructors doesn't work for night as I didn't
# download the data of MOD21A1N.061.

Else
----
@author: Romuald Ait Bachir
"""

import numpy as np
import pandas as pd

from json import load as json_load
from torch.utils.data import Dataset

import utils as us
    
class ModisDatasetB(Dataset):
    """
    Description
    -----------
    Implementation of a torch.utils.data.Dataset for the training of the modelB.
    
    Attributes
    ----------
    path : str
        Path to the .csv dataset.
    transf : str
        Transformation applied to the data. Choose between '-1_1', '0-1' and 'norm'.
        Note: 'norm' refers to Z-score normalization == standardization. 
        The default is 'norm'.
    split : str
        3 different datasets: Train, Val, Test.
    pairs : pandas.DataFrame
        pandas.DataFrame with 3 columns (number, LST, NDVI) containing the path
        to the LST and NDVI tiff files.
    """
    
    def __init__(self, csv_path, transf = 'norm', split = 'Train', time = 'Both'):
        """
        Description
        -----------
        Constructor of the ModisDatasetB class.
        
        Parameters
        ----------
        csv_path : str
            Typical path: "./data/ModisDatasetB.csv"
        split : str, optional
            Choose between "Train", "Test", "Val". The default is 'Train'
        time : str, optional
            Choosing a time of the day. By default, you get both the day and night. ('Both', 'day', 'night')
        """

        self.path = csv_path
        df = pd.read_csv(self.path, sep = ',')
        df.drop(columns=df.columns[0], axis=1, inplace=True)
            
        self.transf = transf
        self.split = split
        
        # Choosing only the split and time wanted
        df = df.loc[df['split'] == self.split]
        if not time == 'Both':
            self.pairs = df.loc[df['LST'].str.contains(time)]
            
        else:
            self.pairs = df
        
        self.stats = json_load(open("./data/statistics.json"))
        
        del df # Just in case the object is not deleted....
        
        
    def __len__(self):
        """
        Description
        -----------
        Method returning the number of element inside the ModisDatasetB.

        Returns
        -------
        len : int
            Length of the dataset.
        """
        
        return len(self.pairs)
    
    
    def __getitem__(self, idx):
        """ 
        Description
        -----------
        Method returning the LST image, an upsampled LST and the associated NDVI
        image located at line idx inside the dataset located in self.pairs.
        
        Parameters
        ----------
        idx : int
            Index of the line read
        
        Returns
        -------
        lst : np.array
            LST image. Shape: (1, 64, 64)
        lst_up : np.array
            LST upsampled image. Shape: (1, 256, 256)
        ndvi : np.array
            NDVI image. Shape: (1, 64, 64)
        """
        
        line_read = self.pairs.iloc[idx]
        lst, _, _, _, _ = us.read_GeoTiff(line_read['LST'])
        ndvi, _, _, _, _ = us.read_GeoTiff(line_read['NDVI'])
        
        if self.transf == '-1_1':
            lst = lst/self.stats['maxi'] # Shrinking by the max temperature lst -> [0,1]
            lst = 2*(lst - 0.5) # lst -> [-1, 1]
            
        elif self.transf == '0-1':
            lst = lst/self.stats['maxi'] # Shrinking by the max temperature lst -> [0,1]
       
        elif self.transf == 'norm':
            # Normalization X_norm = (X - mean)/std
            lst -= self.stats['mean_lst']
            lst /= self.stats['std_lst']
            ndvi -= self.stats['mean_ndvi']
            ndvi /= self.stats['std_ndvi']
        
        lst_up = us.upsampling(lst, (4,4)) 
        return np.expand_dims(lst, axis = 0), np.expand_dims(lst_up, axis = 0), np.expand_dims(ndvi, axis = 0)


class ModisDatasetB_scale_invariance(Dataset):
    """
    Description
    -----------
    Implementation of a torch.utils.data.Dataset for the training of the modelB.
    
    Attributes
    ----------
    path : str
        Path to the .csv dataset.
    transf : str
        Transformation applied to the data. Choose between '-1_1', '0-1' and 'norm'.
        Note: 'norm' refers to Z-score normalization == standardization. 
        The default is 'norm'.
    split : str
        3 different datasets: Train, Val, Test.
    pairs : pandas.DataFrame
        pandas.DataFrame with 3 columns (number, LST, NDVI) containing the path
        to the LST and NDVI tiff files.
    """
    
    def __init__(self, csv_path, transf = 'norm', split = 'Train', time = 'Both'):
        """
        Description
        -----------
        Constructor of the ModisDatasetB class.
        
        Parameters
        ----------
        csv_path : str
            Typical path: "./data/ModisDatasetB.csv"
        split : str, optional
            Choose between "Train", "Test", "Val". The default is 'Train'
        time : str, optional
            Choosing a time of the day. By default, you get both the day and night. ('Both', 'day', 'night')
        """

        self.path = csv_path
        df = pd.read_csv(self.path, sep = ',')
        df.drop(columns=df.columns[0], axis=1, inplace=True)
            
        self.transf = transf
        self.split = split
        
        # Choosing only the split and time wanted
        df = df.loc[df['split'] == self.split]
        if not time == 'Both':
            self.pairs = df.loc[df['LST'].str.contains(time)]
            
        else:
            self.pairs = df
        
        self.stats = json_load(open("./data/statistics.json"))
        
        del df # Just in case the object is not deleted....
        
        
    def __len__(self):
        """
        Description
        -----------
        Method returning the number of element inside the ModisDatasetB.

        Returns
        -------
        len : int
            Length of the dataset.
        """
        
        return len(self.pairs)
    
    
    def __getitem__(self, idx):
        """ 
        Description
        -----------
        Method returning the LST image, an upsampled LST and the associated NDVI
        image located at line idx inside the dataset located in self.pairs.
        
        Parameters
        ----------
        idx : int
            Index of the line read
        
        Returns
        -------
        lst : np.array
            LST image. Shape: (1, 64, 64)
        lst_up : np.array
            LST upsampled image. Shape: (1, 256, 256)
        ndvi : np.array
            NDVI image. Shape: (1, 64, 64)
        """
        
        line_read = self.pairs.iloc[idx]
        lst, _, _, _, _ = us.read_GeoTiff(line_read['LST'])
        ndvi, _, _, _, _ = us.read_GeoTiff(line_read['NDVI'])
        
        if self.transf == '-1_1':
            lst = lst/self.stats['maxi'] # Shrinking by the max temperature lst -> [0,1]
            lst = 2*(lst - 0.5) # lst -> [-1, 1]
            
        elif self.transf == '0-1':
            lst = lst/self.stats['maxi'] # Shrinking by the max temperature lst -> [0,1]
       
        elif self.transf == 'norm':
            # Normalization X_norm = (X - mean)/std
            lst -= self.stats['mean_lst']
            lst /= self.stats['std_lst']
            ndvi -= self.stats['mean_ndvi']
            ndvi /= self.stats['std_ndvi']
        
        # Here we reduce to x4 and to x1 and keep lst as the ground truth. 
        ndvi_1km = us.downscale_LST_SR_to_LR_test(ndvi, deci_type = 'bic').numpy()[0,0,:,:]
        lst_4km = us.downscale_LST_SR_to_LR_test(lst * self.stats['std_lst'] + self.stats['mean_lst'], deci_type = 'norm-L4').numpy()[0,0,:,:]
        
        lst_4km_up = us.upsampling(lst_4km, (4,4)) 
        lst_4km_up = (lst_4km_up - self.stats['mean_lst']) / self.stats['std_lst'] 
        return np.expand_dims(lst_4km_up, axis = 0), np.expand_dims(ndvi_1km, axis = 0), np.expand_dims(lst, axis = 0)
