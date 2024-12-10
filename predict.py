#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:24:52 2024

@author: r24aitba
"""

from model import ModelB_2
import torch
import matplotlib.backends
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pandas as pd
import utils as us
import os
from argparse import ArgumentParser
from json import load as json_load
from osgeo import gdal
import rasterio as rio
import subprocess

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--MOD21A1D_file_path', type=str)
    parser.add_argument('--MOD09GQ_file_path', type=str)
    parser.add_argument('--prediction_device', type=str, default='cpu')
    args = parser.parse_args()
    
    ### Initialization ###
    #Parameters
    lst_path = args.MOD21A1D_file_path # Absolute path preferable
    ndvi_path = args.MOD09GQ_file_path # Absolute path preferable
    prediction_device = args.prediction_device
    save_path = "./predictions"
    
    # Model (for now is set to modelB_1009)
    params_file = './models/modelB_1009/modelB_train_params.json'
    _, _, modelB_parameters, _, save_parameters, device = us.read_JsonB(params_file)
    weights_path = os.path.join(save_parameters['save_path'], save_parameters['model_name'] + '_state_dict.pt')

    modelB = ModelB_2(in_channels = modelB_parameters['in_channels'], 
                      downchannels = modelB_parameters['downchannels'],
                      padding_mode = modelB_parameters['padding_mode'],
                      activation = modelB_parameters['activation'],
                      bilinear = modelB_parameters['bilinear'],
                      n_bridge_blocks = modelB_parameters['n_bridge_blocks']).to(prediction_device)
    
    if device != prediction_device:
        try:
            t = torch.load(weights_path, map_location=torch.device(prediction_device))
            modelB.load_state_dict(t)
        except:
            t = torch.load(weights_path, map_location=torch.device(prediction_device))
            rmv = []
            for key in t:
                if "factor" in key:
                    rmv.append(key)
                    
            for key in rmv:
                del t[key]
                
            modelB.load_state_dict(t)
    else: 
        modelB.load_state_dict(torch.load(weights_path))
  
    modelB.eval()
    
    stats = json_load(open('./data/statistics.json'))
    
    ### Super-resolution ###
    # I suppose in this script that the input files are the .hdf. What I want is for 
    # - a first approach to predict the super resolution of a whole modis .hdf by doing
    # block prediction. 
    LST_K_day, LST_QC, cols, rows, projection, geotransform = us.read_LST(lst_path, 'day')
    Red, NIR, cols, rows, projection, geotransform = us.read_NIRRED(ndvi_path)
    NDVI = us.compute_NDVI(NIR, Red)
    
    # 1st approach: Block super resolution approach.
    LST_SR = np.zeros((NDVI.shape[0], NDVI.shape[1]))
    window_size = 64
    
    for i in range(0, LST_K_day.shape[0], window_size):
        for j in range(0, LST_K_day.shape[1], window_size):
            lst_block = LST_K_day[i : i + window_size, j : j + window_size]
            ndvi_block = NDVI[4*i : 4*(i + window_size), 4*j : 4*(j + window_size)]
            ndvi_block[ndvi_block>1] = 1
            ndvi_block[ndvi_block<-1] = -1
            
            temp_cond = np.zeros((lst_block.shape[0], lst_block.shape[1]))
            temp_cond[lst_block == 0.0] = 1
            
            # Condition on the coverage
            if np.sum(np.sum(temp_cond)) <= window_size ** 2 and lst_block.shape[0] == 64 and lst_block.shape[1] == 64:
                lst =  (lst_block - stats['mean_lst']) / stats['std_lst']
                lst_up_t = (torch.tensor(us.upsampling(lst, (4,4))).unsqueeze(0).unsqueeze(0))
                
                ndvi_t = (torch.tensor(ndvi_block).unsqueeze(0).unsqueeze(0) - stats['mean_ndvi']) / stats['std_ndvi']
                with torch.inference_mode():
                    lst_sr = modelB(torch.cat((lst_up_t, ndvi_t), dim = 1)).numpy()[0,0,:,:] * stats['std_lst'] + stats['mean_lst']
                
                LST_SR[4*i : 4*(i + window_size), 4*j : 4*(j + window_size)] = lst_sr
            
    # Finally, we save it into the predictions folder.
    # In order to keep the metadata, we'll extract it from a subdataset of a product.
    ds_name = 'HDF4_EOS:EOS_GRID:'+ ndvi_path +':MODIS_Grid_2D:sur_refl_b01_1'
    command = "gdal_translate -of GTiff {} {}".format(ds_name, os.path.join(save_path, 'tmp.tiff'))
    subprocess.run(command, shell = True)
    
    f = rio.open(os.path.join(save_path, 'tmp.tiff'))
    
    # Saving the prediction
    # NEED TO EXTRACT THE PREDICTION IN ORDER TO FORMAT THE NAME.
    with rio.open(
        os.path.join(save_path, 'prediction.tiff'),
        'w',
        driver = 'GTiff',
        height = LST_SR.shape[0],
        width = LST_SR.shape[1],
        count=1,
        dtype = LST_SR.dtype,
        crs = f.crs,
        transform = f.transform
    ) as dst:
        dst.write(LST_SR, 1)

    os.remove(os.path.join(save_path, 'tmp.tiff'))
