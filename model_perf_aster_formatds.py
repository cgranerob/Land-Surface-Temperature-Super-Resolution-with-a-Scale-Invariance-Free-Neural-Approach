#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Script to run in order to evaluate the performance of a model. 
It looks for the .hdf or .tif files inside ./test_data corresponding to the LST 
(for MODIS and ASTER) and the NDVI in order to the super resolution and the reprojection.

In the end, this script will produce: 
    - Save the average performances for PSNR, SSIM and RMSE at the bottom of performances.csv (in ./test_data_formatted/results/performances.csv)
    - Save the "end_of_pipeline" data (in ./test_data_formatted/results/{idx}_dict_pred.pkl)
    
Example
-------
To run this script, you can probably do: 
>>> python model_performance_aster.py

or you can just use the execute button inside spyder.

Else
----
@author: Romuald Ait Bachir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import skimage.metrics as met
import torch
import scipy as sp

import subprocess
import os
import pickle

from osgeo import gdal
import rasterio as rio
from rasterio.enums import Resampling
from shapely.geometry import box
from lpips import LPIPS
import data_mining_sharpener_modified as dms

from model import ModelB_2
import utils as us

pd.set_option("display.precision", 2)
#%% The top paths are probably irrelevant with the formatted test dataset
path_formatted_dataset = './test_data_formatted'
path_formatted_data = os.path.join(path_formatted_dataset, "data")
path_formatted_results = os.path.join(path_formatted_dataset, "results")
path_formatted_temporary = os.path.join(path_formatted_dataset, 'tmp')

os.makedirs(path_formatted_temporary, exist_ok = True)

# Loading the dataset dictionnary.
df = pd.read_csv(os.path.join(path_formatted_dataset, "dataset.csv"))
df.drop(columns = "Unnamed: 0", inplace = True)
dict_files = df.T.to_dict(orient= 'list')

dict_results = {}

model_info = './models/modelB_1009/modelB_train_params.json' # SIF-NN-SR1
# model_info =  './models/modelB_2609/modelB_train_params.json' # SIF-NN-SR2
# model_info =  './models/modelB_2011/modelB_train_params.json' # SC-Unet with scale invariance ie learned 4km --> 1km

# To use the scale invariance model, need to uncomment commented lines in if 
# statement of modelB inside the with torch inference mode and comment the lines coming directly 
# after it.

dataset_parameters, _, modelB_parameters, _, save_parameters, device = us.read_JsonB(model_info)
weights_path = os.path.join(save_parameters['save_path'], save_parameters['model_name'] + '_state_dict.pt')

stats = us.json_load(open('./data/statistics.json'))

prediction_device = 'cpu'

sr_type = "modelB" # Choose between: modelB, bicubic, TsHARP, ATPRK, AATPRK, DMS

# Making the folders where the results and the output data will be saved.
if sr_type != "modelB":
    # figures path
    path_results_model = os.path.join(path_formatted_results, sr_type, 'figures')
    os.makedirs(path_results_model, exist_ok = True)
    
    # output path
    path_output_model = os.path.join(path_formatted_results, sr_type)
    os.makedirs(path_output_model, exist_ok = True)
    
else:
    # figures path
    path_results_model = os.path.join(path_formatted_results, model_info.split('/')[-2], 'figures')
    os.makedirs(path_results_model, exist_ok = True)
    
    # output path
    path_output_model = os.path.join(path_formatted_results, model_info.split('/')[-2])
    os.makedirs(path_output_model, exist_ok = True)


# Loading the model based on the type of inference (only if we use it).
if sr_type == "modelB":

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

#%%

lpips_loss = LPIPS(distance = 'mse', reduction = 'mean', mean = [0.0,0.0,0.0], std = [1.0,1.0,1.0]) # Initializing the LPIPS object
grad_lst_aster_tot = [] # In this list, we'll concatenate the gradients of Aster in order 
# to have estimation over the whole dataset.

for idx in dict_files.keys():
    print(idx)
    
    # Loading the data from the csv
    aster_tif = dict_files[idx][0]
    modis_dict_path = dict_files[idx][1]
    
    with open(modis_dict_path, 'rb') as f:
        modis_dict = pickle.load(f)
    
    # Extracting the information from the dictionnary
    lst = modis_dict['LST']
    ndvi = modis_dict['NDVI']
    center_lst = modis_dict["center_lst"] 
    center = modis_dict["center_ndvi"]
    f1_crs = modis_dict["CRS"] # Similar to MODIS one
    transform_sr = modis_dict["transform affine SR"]
    aster_srs = modis_dict["to CRS"] 
    angle = modis_dict["aster_angle"]
    geotransform_lst_patch = modis_dict['geo LST']
    geotransform_ndvi_patch = modis_dict['geo NDVI']
    proj_NDVI = modis_dict['proj NDVI']
    
    # Defining the projections and the set parameters
    modis_srs = '"+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"'
    interp = 'bilinear'
    
    # Computing ndvi_down 
    if ndvi.shape[0] == 256 and ndvi.shape[1] == 256:
        ndvi_down = us.downsampling_img(ndvi, (4, 4)) 
        ndvi_down[ndvi_down>1] = 1
        ndvi_down[ndvi_down<-1] = -1
    else: 
        continue
    
    lst_low = np.zeros((256,256))
    for i in range(lst.shape[0]):
        for j in range(lst.shape[1]):
            lst_low[4*i:4*(i+1), 4*j:4*(j+1)] = lst[i,j]
    
    # Works for bicubic directly since we give the bicubic as input of the model
    lst_sr = cv2.resize(lst, dsize = (256, 256), interpolation=cv2.INTER_CUBIC)
    
    if sr_type == 'modelB':
        lst_sr = torch.tensor((lst_sr - stats['mean_lst']) / stats['std_lst'], dtype = torch.float32)
        ndvi = torch.tensor((ndvi - stats['mean_ndvi']) / stats['std_ndvi'], dtype = torch.float32)
        lst_sr = lst_sr.unsqueeze(0).unsqueeze(0)
        ndvi = ndvi.unsqueeze(0).unsqueeze(0)
        
        x_lst_ndvi = torch.cat((lst_sr, ndvi), dim = 1)
        
        with torch.inference_mode():
            lst_sr = modelB(x_lst_ndvi) # for modelb_2
            
            # # For testing with the scale invariance hypothesis:
            # # Input dimension = 64x64
            # lst_sr_pred = torch.zeros((1,1,256,256))
            # for i in range(4):
            #     for j in range(4):
            #         x_lst_ndvi_piece = x_lst_ndvi[:,:,i*64:(i+1)*64,j*64:(j+1)*64]
            #         lst_sr_pred[:,:,i*64:(i+1)*64,j*64:(j+1)*64] = modelB(x_lst_ndvi_piece)
            
            # lst_sr = lst_sr_pred
            
        lst_sr = lst_sr * stats['std_lst'] + stats['mean_lst']
        lst_sr = lst_sr.numpy()[0,0,:,:]
        
    if sr_type == 'TsHARP':
        
        lst_sr = us.TsHARP(lst, ndvi_down, ndvi, 4, min_T = 273)
    
    # May change the block size from 5 to 4 for both A and AA since 4 represent the 
    # size of fine pixels into a coarse pixels.
    # Doesn't work with 4 so continue with 5.
    if sr_type == 'ATPRK':
        
        lst_sr = us.ATPRK(lst, ndvi_down, ndvi, scale = 4, scc = 926, block_size = 5, min_T = 273)
        
    if sr_type == 'AATPRK':
        
        lst_sr = us.AATPRK(lst, ndvi_down, ndvi, scale = 4, scc = 926, block_size = 5, min_T = 273)
        
    if sr_type == 'DMS': # The DMS implementation on a patch doesn't work fully in comparison to 
    # using the full image. 
        
        lst_name_patch_tif = os.path.join(path_formatted_temporary, "{}_lst_patch.tif".format(idx))
        ndvi_name_patch_tif = os.path.join(path_formatted_temporary, "{}_ndvi_patch.tif".format(idx))
        
        us.save_GeoTiff(lst, lst_name_patch_tif, proj_NDVI, geotransform_lst_patch)
        us.save_GeoTiff(ndvi, ndvi_name_patch_tif, proj_NDVI, geotransform_ndvi_patch)
        
        # We're only using the decision tree
        commonOpts = {"highResFiles":               [ndvi_name_patch_tif],
                      "lowResFiles":                [lst_name_patch_tif],
                      "lowResQualityFiles":         [],
                      "lowResGoodQualityFlags":     [255],
                      "cvHomogeneityThreshold":     0,
                      "movingWindowSize":           0, #Only way to work with patches == removing local regs
                      "disaggregatingTemperature":  True}
        dtOpts =     {"perLeafLinearRegression":    True,
                      "linearRegressionExtrapolationRatio": 0.25}
        opts = commonOpts.copy()
        opts.update(dtOpts)
        disaggregator = dms.DecisionTreeSharpener(**opts)
        
        print("Training regressor...")
        disaggregator.trainSharpener()
        print("Sharpening...")
        downscaledFile, regs = disaggregator.applySharpener(ndvi_name_patch_tif, lst_name_patch_tif) 
        # regs was added to observe the decision tree.
        print("Residual analysis...")
        residualImage, correctedImage = disaggregator.residualAnalysis(downscaledFile, lst_name_patch_tif, doCorrection=True)
        lst_sr = correctedImage.GetRasterBand(1).ReadAsArray(0,0,256,256)

    # Save here LST_low but without nn_upsampling in order to compare to ASTER. Do the same with MOD21A2
    tmp = np.zeros((4800,4800)) # lst_sr
    tmp1 = np.zeros((4800,4800)) # lst_low
    tmp2 = np.zeros((4800,4800)) # ndvi
    tmp1[center[0]-128:center[0]+128, center[1]-128:center[1]+128] = lst_low
    tmp[center[0]-128:center[0]+128, center[1]-128:center[1]+128] = lst_sr
    if sr_type == 'modelB':
        tmp2[center[0]-128:center[0]+128, center[1]-128:center[1]+128] = ndvi.numpy()[0,0,:,:] * stats['std_ndvi'] + stats['mean_ndvi']
    else: 
        tmp2[center[0]-128:center[0]+128, center[1]-128:center[1]+128] = ndvi
    
    # Setting the path of the patches
    path_lst_sr = os.path.join(path_formatted_temporary, "{}_lst_sr_modis.tif".format(idx))
    path_lst_lr = os.path.join(path_formatted_temporary, "{}_lst_modis.tif".format(idx))
    path_ndvi = os.path.join(path_formatted_temporary, "{}_ndvi_modis.tif".format(idx))
    
    with rio.open(
        path_lst_sr,
        'w',
        driver='GTiff',
        height=tmp.shape[0],
        width=tmp.shape[1],
        count=1,
        dtype=tmp.dtype,
        crs=f1_crs,
        transform=transform_sr,
    ) as dst:
        dst.write(tmp, 1)
        
    with rio.open(
        path_lst_lr,
        'w',
        driver='GTiff',
        height=tmp1.shape[0],
        width=tmp1.shape[1],
        count=1,
        dtype=tmp1.dtype,
        crs=f1_crs,
        transform=transform_sr,
    ) as dst:
        dst.write(tmp1, 1)
    
    with rio.open(
        path_ndvi,
        'w',
        driver='GTiff',
        height=tmp2.shape[0],
        width=tmp2.shape[1],
        count=1,
        dtype=tmp2.dtype,
        crs=f1_crs,
        transform=transform_sr,
    ) as dst:
        dst.write(tmp2, 1)
    
    # Reprojecting the LST_SR, NDVI and the LST_LR into the UTM representation.
    path_lst_sr_reproj = os.path.join(path_formatted_temporary, "{}_lst_sr_modis_reproj.tif".format(idx))
    path_lst_lr_reproj = os.path.join(path_formatted_temporary, "{}_lst_modis_reproj.tif".format(idx))
    path_ndvi_reproj = os.path.join(path_formatted_temporary, "{}_ndvi_modis_reproj.tif".format(idx))
    
    command2_lst = "gdalwarp -s_srs {} -t_srs {} -r {} {} {}".format(modis_srs, aster_srs, interp, path_lst_sr, path_lst_sr_reproj)
    command2_lst_lr = "gdalwarp -s_srs {} -t_srs {} -r {} {} {}".format(modis_srs, aster_srs, interp, path_lst_lr, path_lst_lr_reproj)
    command2_ndvi = "gdalwarp -s_srs {} -t_srs {} -r {} {} {}".format(modis_srs, aster_srs, interp, path_ndvi, path_ndvi_reproj)
    subprocess.run(command2_lst, shell = True)
    subprocess.run(command2_lst_lr, shell = True)
    subprocess.run(command2_ndvi, shell = True)
    
    ###### FINALLY, WE FIND THE COMMON AREA BETWEEN IN OUR DATA.
    # ras_1 = os.path.join(aster_250m, "{}_reproj_aster_250m.tif".format(idx))  # Aster 250m
    ras_1 = aster_tif
    ras_2 = path_lst_sr_reproj # MOD21A1D super resolution
    ras_3 = path_lst_lr_reproj # MOD21A1D original
    ras_4 = path_ndvi_reproj # Corresponding NDVI
    
    with rio.open(ras_1) as ras1, rio.open(ras_2) as ras2, rio.open(ras_3) as ras3, rio.open(ras_4) as ras4: # Without the RGB
        transform_int = ras1.transform * ras1.transform.shear(angle, -angle)
        
        # NOTE: SOMETHING WEIRD SEEMS To HAPPEN ON THE ANGLE...                
        x_scale = transform_int.a / ras2.transform.a
        y_scale = transform_int.e / ras2.transform.e
        
        transform = ras1.transform * ras1.transform.scale(
            (ras1.width / ras1.shape[-1]),
            (ras1.height / ras1.shape[-2])
        )
        
        ### For Aster
        ext1 = box(*ras1.bounds)
        ext2 = box(*ras2.bounds)
        intersection = ext1.intersection(ext2)
        win1 = rio.windows.from_bounds(*intersection.bounds, transform)
        win2 = rio.windows.from_bounds(*intersection.bounds, ras2.transform)
        
        # The window should be the same for all the files coming from Modis
        overlap_1 = ras1.read(window=win1,
        out_shape=(
            ras1.count,
            int(win1.height * y_scale),
            int(win1.width * x_scale)
        ),
        resampling=Resampling.bilinear)
        overlap_2 = ras2.read(window = win2)
        overlap_3 = ras3.read(window = win2)
        overlap_4 = ras4.read(window = win2)
        
    # Applying the scale factor
    overlap_1 *= 0.1
    
    p1_aster,p2_aster = us.find_corners(overlap_1[0], us.condition_lst_aster)
    p1_modis,p2_modis = us.find_corners(overlap_2[0], us.condition_lst_modis)
    
    p1 = (max(p1_modis[0], p1_aster[0]) + 1, min(p1_modis[1], p1_aster[1]) - 1 )
    p2 = (max(p2_modis[0], p2_aster[0]) + 1, min(p2_modis[1], p2_aster[1]) - 1 )

    overlap_11 = overlap_1.copy()[0,p2[0]-1:p2[1]-1,p1[0]-1:p1[1]-1] 
    overlap_22 = overlap_2.copy()[0,p2[0]-1:p2[1]-1,p1[0]-1:p1[1]-1]
    overlap_33 = overlap_3.copy()[0,p2[0]-1:p2[1]-1,p1[0]-1:p1[1]-1]
    overlap_44 = overlap_4.copy()[0,p2[0]-1:p2[1]-1,p1[0]-1:p1[1]-1]

    if overlap_11.shape[0] > 40 and overlap_11.shape[1] > 40:
    
        maxi = np.max([overlap_11,overlap_22])
        mini = np.min([overlap_11,overlap_22])
        
        # Computing the rmse for low and high ASTER grads
        aster_lst_1000m = us.get_output_ftm(torch.tensor(overlap_11).unsqueeze(0).unsqueeze(0)).numpy()[0,0,:,:]
        grad_aster = np.abs(overlap_11 - aster_lst_1000m)
        
        grad_lst_aster_tot += list(grad_aster.flatten())
        
        sqe_modis_sr_aster = np.power(overlap_11 - overlap_22, 2)
        
        grad_aster_low = np.percentile(grad_aster.flatten(), 25)
        grad_aster_high = np.percentile(grad_aster.flatten(), 75)
        
        error_low_grad = sqe_modis_sr_aster.copy()
        error_low_grad[grad_aster >= grad_aster_low] = 0
        vals_without_zeros = list(filter((0.0).__ne__, list(error_low_grad.flatten())))
        rmse_low_grad = np.sqrt(np.mean(vals_without_zeros))
        
        error_mean_grad = sqe_modis_sr_aster.copy()
        error_mean_grad[grad_aster < grad_aster_low] = 0
        error_mean_grad[grad_aster > grad_aster_high] = 0
        vals_without_zeros = list(filter((0.0).__ne__, list(error_mean_grad.flatten())))
        rmse_mean_grad = np.sqrt(np.mean(vals_without_zeros))
        
        error_high_grad = sqe_modis_sr_aster.copy()
        error_high_grad[grad_aster < grad_aster_high] = 0
        vals_without_zeros = list(filter((0.0).__ne__, list(error_high_grad.flatten())))
        rmse_high_grad = np.sqrt(np.mean(vals_without_zeros))
        # The 0.0 is really important. If only using 0, the filter
        # doesn't work

        # Computing LPIPS
        # Making the data in right format
        t1 = torch.tensor((overlap_11 - mini)/(maxi - mini), dtype = torch.float).repeat(1,3,1,1)
        t2 = torch.tensor((overlap_22 - mini)/(maxi - mini), dtype = torch.float).repeat(1,3,1,1)
        
        val_lpips = lpips_loss(t1, t2).numpy()
        
        # Computing the error on the gradients
        sobels = [ [[1,2,1],[0,0,0],[-1,-2,-1]],
             [[1,0,-1],[2,0,-2],[1,0,-1]],
             [[2,1,0],[1,0,-1],[0,-1,-2]],
             [[0,1,2],[-1,0,1],[-2,-1,0]]]
    
        grads_modis = []
        grads_aster = []
        for sobel in sobels:
            grads_modis.append(sp.signal.convolve2d(overlap_22, sobel, mode = 'valid'))
            grads_aster.append(sp.signal.convolve2d(overlap_11, sobel, mode = 'valid'))
    
        mag_modis = np.sqrt(np.power(grads_modis[0],2) +np.power(grads_modis[1],2) +np.power(grads_modis[2],2) + np.power(grads_modis[3],2))
        mag_aster = np.sqrt(np.power(grads_aster[0],2) +np.power(grads_aster[1],2) +np.power(grads_aster[2],2) + np.power(grads_aster[3],2))
        
        
        dict_results[idx] = [
        met.peak_signal_noise_ratio(overlap_11, overlap_22, data_range=maxi - mini),
        met.structural_similarity(overlap_11, overlap_22, data_range=maxi - mini),
        np.sqrt(np.mean(sqe_modis_sr_aster)),
        rmse_low_grad,
        rmse_mean_grad,
        rmse_high_grad,
        us.gssim(overlap_11, overlap_22, data_range = maxi-mini),
        val_lpips,
        np.sqrt(np.mean((np.power(mag_modis - mag_aster,2))))
        ]
        
        # For plotting the predictions
        fig = plt.figure(1, figsize = (16, 14))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.5)
           
        ax1 = fig.add_subplot(gs[:2, :2])
        ig1 = ax1.imshow(overlap_3[0,:,:], cmap = 'jet', aspect = 'auto', vmin = mini, vmax = maxi)
        plt.colorbar(ig1)
        ax1.title.set_text("LST MODIS")
        
        ax2 = fig.add_subplot(gs[:2, 2:])
        ig2 = ax2.imshow(overlap_4[0,:,:], cmap = 'RdYlGn', aspect = 'auto')
        plt.colorbar(ig2)
        ax2.title.set_text('NDVI')
           
        ax3 = fig.add_subplot(gs[2:, :2])
        ig3 = ax3.imshow(overlap_2[0,:,:], cmap = 'jet', aspect = 'auto', vmin = mini, vmax = maxi)
        plt.colorbar(ig3)
        ax3.title.set_text("MODIS LST SR")
           
        ax4 = fig.add_subplot(gs[2:, 2:])
        ig4 = ax4.imshow(overlap_1[0,:,:], cmap = 'jet', aspect = 'auto', vmin = mini, vmax = maxi) 
        plt.colorbar(ig4)
        ax4.title.set_text('ASTER LST')
        
        plt.savefig(os.path.join(path_results_model, '{}_predictions.png').format(idx))
        plt.close(fig)
        plt.close('all')
        
        # Plotting the final cropped results
        fig = plt.figure(2, figsize = (16, 14))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.5)
        
        ax1 = fig.add_subplot(gs[:2, :2])
        ig1 = ax1.imshow(overlap_33, cmap = 'jet', aspect = 'auto', vmin = mini, vmax = maxi)
        plt.colorbar(ig1)
        ax1.title.set_text("LST MODIS")
        
        ax2 = fig.add_subplot(gs[:2, 2:])
        ig2 = ax2.imshow(overlap_44, cmap = 'RdYlGn', aspect = 'auto')
        plt.colorbar(ig2)
        ax2.title.set_text('NDVI')
           
        ax3 = fig.add_subplot(gs[2:, :2])
        ig3 = ax3.imshow(overlap_22, cmap = 'jet', aspect = 'auto', vmin = mini, vmax = maxi)
        plt.colorbar(ig3)
        ax3.title.set_text("MODIS LST SR")
           
        ax4 = fig.add_subplot(gs[2:, 2:])
        ig4 = ax4.imshow(overlap_11, cmap = 'jet', aspect = 'auto', vmin = mini, vmax = maxi) 
        plt.colorbar(ig4)
        ax4.title.set_text('ASTER LST')
        
        plt.savefig(os.path.join(path_results_model, '{}_predictions_cropped.png').format(idx))
        plt.close(fig)
        plt.close('all')
        
        # Finally, we save the prediction and the LST aster for 
        dict_output = {'LST:': overlap_33,
                       'NDVI': overlap_44,
                       'LST_ASTER': overlap_11,
                       'LST_SR': overlap_22}
        with open(os.path.join(path_output_model, '{}_dict_pred.pkl'.format(idx)), 'wb') as f:
            pickle.dump(dict_output, f)

# Saving the results
df_results = pd.DataFrame.from_dict(dict_results, columns = ('PSNR', 'SSIM', 'RMSE', 'RMSE (low grad per image)', 'RMSE (mean grad per image)', 'RMSE (high grad per image)', 'GSSIM', 'LPIPS', 'RMSE_grad'), orient = 'index')

# Displaying the performances
df1_results = df_results.loc[df_results['PSNR'] != 'NA']
print("|-------- Statistics --------|")
print(df1_results.mean())

df_results.loc['mean'] = df1_results.mean()
df_results.loc['std'] = df1_results.std()
df_results.loc['10%'] = df1_results.quantile(0.1)
df_results.loc['Q1'] = df1_results.quantile(0.25)
df_results.loc['mediane'] = df1_results.quantile(0.5)
df_results.loc['Q3'] = df1_results.quantile(0.75)
df_results.loc['90%'] = df1_results.quantile(0.9)

if sr_type == 'modelB':
    df_results.to_csv(os.path.join(path_output_model,'performances.csv'))
else:
    df_results.to_csv(os.path.join(path_output_model,'performances.csv'))
    
# Removing the intermediary generated files
files_to_remove = [os.path.join(path_formatted_temporary, f) for f in os.listdir(path_formatted_temporary) if os.path.isfile(os.path.join(path_formatted_temporary, f))] 

for file in files_to_remove:
    os.remove(file)
