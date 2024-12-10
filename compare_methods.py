#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Code to use as many individual cells serving different purposes, mostly centered
on results analysis, making figures, discussions.

It is made to be ran on the results coming from model_performance_aster as 
model_perf_aster_format.py cant be used to predict DMS which is required for all the 
visualizations.

# LOOK LINE ~410 if you want to generate all the figures for the whole dataset. 

Else
----
@author: Romuald Ait Bachir
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset import ModisDatasetB
import torch

import pickle
import os
import utils as us

plt.rcParams.update({
    "text.usetex": True,})
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}')

os.makedirs('./figures_test_dataset', exist_ok = "True")

#%% Observing what is the distribution of the files along the seasons in the train dataset.
# It could be interesting to see the performances on ASTER per season.

ds = ModisDatasetB("./data/ModisDatasetB.csv", transf = 'norm', split = 'Train', time = 'day')
data = np.array(ds.pairs['LST'].to_numpy(), dtype = str)
day_list = [int(i.split('.')[-6][5:]) for i in data]

#plt.hist(day_list, bins=365)

start_winter = 335
end_winter = 59

start_spring = 60
end_spring = 151

start_summer = 152
end_summer = 243

start_fall = 244
end_fall = 334

dict_seasons = {"Winter": 0,
                "Spring": 0,
                "Summer": 0,
                "Fall"  : 0 }

for day in day_list: 
    if day > start_winter or day<end_winter: 
        dict_seasons["Winter"] += 1
    if day > start_spring and day<end_spring: 
        dict_seasons["Spring"] += 1
    if day > start_summer and day<end_summer: 
        dict_seasons["Summer"] += 1
    if day > start_fall and day<end_fall: 
        dict_seasons["Fall"] += 1

plt.bar(list(dict_seasons.keys()), dict_seasons.values(), color='r')
plt.show()

#%% Observing how the relationship between the LST and NDVI modis.
ds = ModisDatasetB("./data/ModisDatasetB.csv", transf = 'norm', split = 'Train', time = 'day')

slope_list = []
intercept_list = []
error_reg_lin = []
error_reg_tree = []
reg_tree_list = []

cnt = 0
fig = plt.figure(1)
for idx in range(len(ds)):
    if not cnt%1000:
        print(cnt)
        
    data = ds[idx]
    lst = data[0][0,:,:]
    ndvi = data[2][0,:,:]
    lst_bic = data[1][0,:,:]
    
    lst_nn = np.zeros((ndvi.shape[0],ndvi.shape[1]))
    for i in range(lst.shape[0]):
        for j in range(lst.shape[1]):
            lst_nn[4*i:4*(i+1),4*j:4*(j+1)] = lst[i,j]
    
    # Linear regression
    reg = sp.stats.linregress(ndvi.flatten(), lst_nn.flatten())
    slope_list.append(reg.slope)
    intercept_list.append(reg.intercept)
    prediction = reg.slope * ndvi + reg.intercept
    
    error = np.sqrt(np.mean((lst_nn - prediction)**2))
    error_reg_lin.append(error)
    
    # Plotting for the linear regression
    if cnt < 9:
        ax = plt.subplot(3,3,cnt+1)
        cnt += 1
        plt.hexbin(ndvi.flatten(), lst_nn.flatten(), gridsize=120, cmap ='jet')
        plt.plot([min(ndvi.flatten()), max(ndvi.flatten())], [reg.slope*min(ndvi.flatten())+reg.intercept, reg.slope*max(ndvi.flatten())+reg.intercept], color = 'red')
        plt.grid()
        plt.xlim([min(ndvi.flatten()),max(ndvi.flatten())])
        plt.ylim([min(lst_nn.flatten()),max(lst_nn.flatten())])
        fig.tight_layout()
        if reg.intercept < 0:
            ax.title.set_text("LST"+' = {0:.2g}NDVI'.format(reg.slope) + "{0:.2g}".format(reg.intercept)+", R² = {0:.2g}".format(reg.rvalue**2))
        else:
            ax.title.set_text("LST"+' = {0:.2g}NDVI'.format(reg.slope) + "+{0:.2g}".format(reg.intercept)+", R² = {0:.2g}".format(reg.rvalue**2))
    
print('Average Regression Parameters')
print('Slope: mu, sigma')
print(np.mean(slope_list))
print(np.std(slope_list))
print('Intercept: mu, sigma')
print(np.mean(intercept_list))
print(np.std(intercept_list))

print('Average RMSE Linear')
print(np.mean(error_reg_lin))
print(np.std(error_reg_lin))

#%% ########################################################################
# Measuring the statistics of the time difference associated only to the predictions
#!!!! NOTE: Uses the non formatted test dataset's informations as the time difference 
# is only stored in it.
path_data = "./test_data"
# path_data = "./test_data2"

df = pd.read_csv(os.path.join(path_data,"aster_modis_dataset_2.csv"), index_col = 0)
df.index = pd.to_numeric(df.index, errors='coerce')

path_results = os.path.join(path_data, 'results/modelB_1009') # Will be used to localize the dataset
performances_df = pd.read_csv(os.path.join(path_results, "performances.csv"))

df_join = pd.merge(df, performances_df, left_index=True, right_index=True)
df_join.dropna(inplace = True)

time_differences = 60 * np.array(df_join.to_numpy()[:,9], dtype = np.float32) 

print("---- Statistics ----")
print("Mean: {:.1f} min".format(np.mean(time_differences)))
print("Std: {:.1f} min".format(np.std(time_differences)))
print("Mini: {:.1f} min".format(np.min(np.abs(time_differences))))
print("Maxi: {:.1f} min".format(np.max(np.abs(time_differences))))

# ASTER is in GMT while MODIS is in SOLAR LOCAL TIME so a transformation was applied following:
# https://gis.stackexchange.com/questions/354643/converting-local-solar-time-to-utc-for-modis
# This conversion was done in create_test_dataset.py.

#%% ########################################################################
# Observing the evolution of the metrics alongside the bias at 1km
# Loading the bias-rmse at 1km, the RMSE at 1km and the time difference at 1km. 
# It is obtained for CNN1 on the first test dataset.
#!!!! NOTE: Uses the non formatted test dataset's results as the error at 1km is 
# only computed in it. (It is also possible to compute it on the formatted test dataset but not 
# enough time to do it)
df_1km = pd.read_csv('./test_data/results/Modis-Aster/bias_aster_modis.csv', index_col = 0)

# Loading the performances of our model
df_sr = pd.read_csv('./test_data/results/modelB_1009/performances.csv', index_col = 0)
df_sr = df_sr.dropna()

df_tot = pd.merge(df_1km, df_sr, left_index = True, right_index = True)
df_tot = df_tot.drop(['mean', 'std'])
df_tot.index = pd.to_numeric(df_tot.index, errors='coerce')

# Loading the dataset containing the time
path_data = "./test_data/"
df = pd.read_csv(os.path.join(path_data,"aster_modis_dataset_2.csv"), index_col = 0)
df.index = pd.to_numeric(df.index, errors='coerce')

df_full = pd.merge(df_tot, df, left_index = True, right_index = True)

data = df_full.to_numpy()

# Variations following the Bias between MOD21A1D and ASTER
plt.figure(1, figsize = (21,14))
plt.subplot(2,3,1)
plt.scatter(np.absolute(data[:,1]),data[:,6])
plt.xlabel('Bias (absolute) MOD21-ASTER')
plt.ylabel('RMSE ASTER 250m-MOD21_SR')

plt.subplot(2,3,2)
plt.scatter(np.absolute(data[:,1]),data[:,4])
plt.xlabel('Bias (absolute) MOD21-ASTER')
plt.ylabel('PSNR ASTER 250m-MOD21_SR')

plt.subplot(2,3,3)
plt.scatter(np.absolute(data[:,1]),data[:,5])
plt.xlabel('Bias (absolute) MOD21-ASTER')
plt.ylabel('SSIM ASTER 250m-MOD21_SR')

plt.subplot(2,3,4)
plt.scatter(np.absolute(data[:,1]),data[:,10])
plt.xlabel('Bias (absolute) MOD21-ASTER')
plt.ylabel('GSSIM ASTER 250m-MOD21_SR')

plt.subplot(2,3,5)
plt.scatter(np.absolute(data[:,1]),data[:,11])
plt.xlabel('Bias (absolute) MOD21-ASTER')
plt.ylabel('LPIPS ASTER 250m-MOD21_SR')

# Variations following the RMSE between MOD21A1D and ASTER
plt.figure(2, figsize = (21,14))
plt.subplot(2,3,1)
plt.scatter(np.absolute(data[:,3]),data[:,6])
plt.xlabel('RMSE (absolute) MOD21-ASTER')
plt.ylabel('RMSE ASTER 250m-MOD21_SR')

plt.subplot(2,3,2)
plt.scatter(np.absolute(data[:,3]),data[:,4])
plt.xlabel('RMSE (absolute) MOD21-ASTER')
plt.ylabel('PSNR ASTER 250m-MOD21_SR')

plt.subplot(2,3,3)
plt.scatter(np.absolute(data[:,3]),data[:,5])
plt.xlabel('RMSE (absolute) MOD21-ASTER')
plt.ylabel('SSIM ASTER 250m-MOD21_SR')

plt.subplot(2,3,4)
plt.scatter(np.absolute(data[:,3]),data[:,10])
plt.xlabel('RMSE (absolute) MOD21-ASTER')
plt.ylabel('GSSIM ASTER 250m-MOD21_SR')

plt.subplot(2,3,5)
plt.scatter(np.absolute(data[:,3]),data[:,11])
plt.xlabel('RMSE (absolute) MOD21-ASTER')
plt.ylabel('LPIPS ASTER 250m-MOD21_SR')

# Variations following the absolute time difference between MOD21A1D and ASTER
plt.figure(3, figsize = (21,14))
plt.subplot(2,3,1)
plt.scatter(60*np.absolute(data[:,24]),data[:,6])
plt.xlabel('Absolute Time diff between MOD21-ASTER')
plt.ylabel('RMSE ASTER 250m-MOD21_SR')

plt.subplot(2,3,2)
plt.scatter(60*np.absolute(data[:,24]),data[:,4])
plt.xlabel('Absolute Time diff between MOD21-ASTER')
plt.ylabel('PSNR ASTER 250m-MOD21_SR')

plt.subplot(2,3,3)
plt.scatter(60*np.absolute(data[:,24]),data[:,5])
plt.xlabel('Absolute Time diff between MOD21-ASTER')
plt.ylabel('SSIM ASTER 250m-MOD21_SR')

plt.subplot(2,3,4)
plt.scatter(60*np.absolute(data[:,24]),data[:,10])
plt.xlabel('Absolute Time diff between MOD21-ASTER')
plt.ylabel('GSSIM ASTER 250m-MOD21_SR')

plt.subplot(2,3,5)
plt.scatter(60*np.absolute(data[:,24]),data[:,11])
plt.xlabel('Absolute Time diff between MOD21-ASTER')
plt.ylabel('LPIPS ASTER 250m-MOD21_SR')

###############################################################################
#%% Implementing the metrics in the Fourier Space defined in Julien's paper
#  Computing the average spectral metrics for the the formatted dataset

path_dataset = "./test_data_formatted"
# path_dataset = "./test_data2_formatted"

# We load the dictionnary of results for bicubic only to extract the indices
df = pd.read_csv(os.path.join(path_dataset, 'results','bicubic', "performances.csv"))
keys_list = np.array(df['Unnamed: 0'].to_numpy()[:-7], dtype = np.int16)

dict_perf = {}
print('--- Computing the spectral similarity metrics ---')
for image_index in keys_list: 
    print(image_index)
    # Choosing the approaches
    models = ['bicubic','TsHARP', 'ATPRK', 'DMS', 'modelB_2011', 'modelB_1009', 'modelB_2609']
    labels = ['bicubic','TsHARP', 'ATPRK', 'DMS', 'SC-Unet', 'SIF-NN-SR1', 'SIF-NN-SR2']
    
    # models = ['bicubic', 'CNN1_alpha_09', 'CNN1_alpha_095', 'CNN1_alpha_096', 'CNN1_alpha_097', 'CNN1_alpha_098']
    # labels = ['bicubic', 'CNN1_alpha=0.9', 'CNN1_alpha=0.95', 'CNN1_alpha=0.96', 'CNN1_alpha=0.97', 'CNN1_alpha=0.98']
    
    temperature_dict = {}
    fourier_dict = {}
    
    # Loading ASTER
    dict_info_path = os.path.join(path_dataset, 'results/bicubic/{}_dict_pred.pkl'.format(image_index))
    
    with open(dict_info_path, 'rb') as f:
        dict_info = pickle.load(f)
    
    l,c = dict_info['LST_ASTER'].shape[0], dict_info['LST_ASTER'].shape[1]
    temperature_dict["ASTER"] = dict_info['LST_ASTER']

    # Making the fourier data.
    aster_tens = torch.tensor(temperature_dict["ASTER"]).unsqueeze(0).unsqueeze(0)
    aster_hf = aster_tens - us.get_output_ftm(aster_tens, mtf = 0.1)

    fourier_dict["ASTER"] = np.fft.fftshift(np.abs(sp.fft.fft2(temperature_dict["ASTER"])))

    # Loading the predictions
    ## This is probably badly optimized as we repeat the same computation multiple times
    for i, model in enumerate(models):
        # Loading LST_SR
        dict_info_path = os.path.join(path_dataset, 'results/', model, "{}_dict_pred.pkl".format(image_index))
        
        with open(dict_info_path, 'rb') as f:
            dict_info = pickle.load(f)

        temperature_dict[model] = dict_info['LST_SR']
        fourier_dict[model] = np.fft.fftshift(np.abs(sp.fft.fft2(temperature_dict[model])))

    # Now that the loading and processing is done, for simplicity we add ASTER inside of the two lists.
    models.insert(0, 'ASTER')
    labels.insert(0, 'ASTER')

    # Getting the 1D attenuation spectra
    attenuation_spectra_dict = {}
    for i,model in enumerate(models):
        attenuation_spectra_dict[model] = us.compute_2D_attenuation_spectra(fourier_dict[model])

    ## Computing the similarity metrics
    for ds_model in models[1:]:
        
        if not ds_model in dict_perf.keys():
            dict_perf[ds_model] = [[],[],[],[],[],[]]
            print("Create list value")
            
        ## Now let's compute the metrics:
        # For simplicity, let's compute the F* being the Decibel representation.
        ASTER_dB = attenuation_spectra_dict['ASTER']
        bic_dB = attenuation_spectra_dict['bicubic']
        model_dB = attenuation_spectra_dict[ds_model]
        
        dict_perf[ds_model][0].append(us.get_PFR(ASTER_dB, bic_dB))
        dict_perf[ds_model][1].append(us.get_AFR(model_dB, ASTER_dB, bic_dB))
        dict_perf[ds_model][2].append(us.get_FRR(model_dB, ASTER_dB, bic_dB))
        dict_perf[ds_model][3].append(us.get_FRO(model_dB, ASTER_dB, bic_dB))
        dict_perf[ds_model][4].append(us.get_FRU(model_dB, ASTER_dB, bic_dB))
        dict_perf[ds_model][5].append(np.sqrt(np.mean((np.array(model_dB) - np.array(ASTER_dB))**2)))

for key in dict_perf.keys():
    print("{}: PFR: {:.2f}, AFR: {:.2f}, FRR: {:.2f}, FRO: {:.2f}, FRU: {:.2f}, RMSE_ATT: {:.2f}".format(key, np.mean(dict_perf[key][0]),  np.mean(dict_perf[key][1]), np.mean(dict_perf[key][2]), np.mean(dict_perf[key][3]), np.mean(dict_perf[key][4]), np.mean(dict_perf[key][5])))

# Finally, updating the performances.csv files per model.
print('--- Updating performances.csv for every model ---')
models = ['bicubic','TsHARP', 'ATPRK', 'DMS', 'modelB_2011', 'modelB_1009', 'modelB_2609']
for model in models:
    df_performances = pd.read_csv(os.path.join(path_dataset, 'results', model, "performances.csv"))
    dict_performances = df_performances.T.to_dict(orient= 'list')
    dict_performances_new = {}
    
    # Getting the real keys
    key_list = []
    for key in dict_performances.keys():
        key_list.append(dict_performances[key][0])
        dict_performances_new[dict_performances[key][0]] = dict_performances[key][1:]
    
    if not 'FRR' in list(df_performances.columns):
        print(1)
        for i, key in enumerate(key_list):
            print(key)
            if key == 'mean':
                dict_performances_new[key].append(np.mean(dict_perf[model][2])) 
                dict_performances_new[key].append(np.mean(dict_perf[model][3]))
                dict_performances_new[key].append(np.mean(dict_perf[model][4]))
                dict_performances_new[key].append(np.mean(dict_perf[model][5])) 
            elif key == 'std':
                dict_performances_new[key].append(np.std(dict_perf[model][2])) 
                dict_performances_new[key].append(np.std(dict_perf[model][3]))
                dict_performances_new[key].append(np.std(dict_perf[model][4]))
                dict_performances_new[key].append(np.std(dict_perf[model][5]))
            elif key == '10%':
                dict_performances_new[key].append(np.percentile(dict_perf[model][2], 10)) 
                dict_performances_new[key].append(np.percentile(dict_perf[model][3], 10))
                dict_performances_new[key].append(np.percentile(dict_perf[model][4], 10))
                dict_performances_new[key].append(np.percentile(dict_perf[model][5], 10)) 
            elif key == 'Q1':
                dict_performances_new[key].append(np.percentile(dict_perf[model][2], 25)) 
                dict_performances_new[key].append(np.percentile(dict_perf[model][3], 25))
                dict_performances_new[key].append(np.percentile(dict_perf[model][4], 25))
                dict_performances_new[key].append(np.percentile(dict_perf[model][5], 25)) 
            elif key == 'mediane':
                dict_performances_new[key].append(np.percentile(dict_perf[model][2], 50)) 
                dict_performances_new[key].append(np.percentile(dict_perf[model][3], 50))
                dict_performances_new[key].append(np.percentile(dict_perf[model][4], 50))
                dict_performances_new[key].append(np.percentile(dict_perf[model][5], 50))
            elif key == 'Q3':
                dict_performances_new[key].append(np.percentile(dict_perf[model][2], 75)) 
                dict_performances_new[key].append(np.percentile(dict_perf[model][3], 75))
                dict_performances_new[key].append(np.percentile(dict_perf[model][4], 75))
                dict_performances_new[key].append(np.percentile(dict_perf[model][5], 75)) 
            elif key == '90%':
                dict_performances_new[key].append(np.percentile(dict_perf[model][2], 90)) 
                dict_performances_new[key].append(np.percentile(dict_perf[model][3], 90))
                dict_performances_new[key].append(np.percentile(dict_perf[model][4], 90))
                dict_performances_new[key].append(np.percentile(dict_perf[model][5], 90)) 
            else:
                dict_performances_new[key].append(dict_perf[model][2][i])
                dict_performances_new[key].append(dict_perf[model][3][i]) 
                dict_performances_new[key].append(dict_perf[model][4][i])
                dict_performances_new[key].append(dict_perf[model][5][i])
    
        df_performances_new = pd.DataFrame.from_dict(dict_performances_new, orient = 'index', columns = ('PSNR', 'SSIM', 'RMSE', 'RMSE (<25%)', 'RMSE (>25%<75%)', 'RMSE (>75%)', 'GSSIM', 'LPIPS', 'RMSE_grad', 'FRR', 'FRO', 'FRU', 'ATTENUATION_RMSE_ASTER_MODEL'))
        df_performances_new.to_csv(os.path.join(path_dataset, 'results', model, "performances.csv"))

#%% Computing an average spectrum of the attenuation spectra of all images in the dataset.

path_dataset = "./test_data_formatted"
# path_dataset = "./test_data2_formatted"

# We load the dictionnary of results for bicubic only to extract the indices
df = pd.read_csv(os.path.join(path_dataset, 'results','bicubic', "performances.csv"))
keys_list = np.array(df['Unnamed: 0'].to_numpy()[:-7], dtype = np.int16)

dict_perf = {}

list_spectra = []
list_x_axis = []

print('--- Computing the spectral similarity metrics ---')
for image_index in keys_list: 
    print(image_index)
    # Choosing the approaches
    models = ['bicubic','TsHARP', 'ATPRK', 'DMS', 'modelB_2011', 'modelB_1009', 'modelB_2609']
    labels = ['bicubic','TsHARP', 'ATPRK', 'DMS', 'SC-Unet', 'SIF-NN-SR1', 'SIF-NN-SR2']
    
    # models = ['bicubic', 'CNN1_alpha_09', 'CNN1_alpha_095', 'CNN1_alpha_096', 'CNN1_alpha_097', 'CNN1_alpha_098']
    # labels = ['bicubic', 'CNN1_alpha=0.9', 'CNN1_alpha=0.95', 'CNN1_alpha=0.96', 'CNN1_alpha=0.97', 'CNN1_alpha=0.98']
    
    temperature_dict = {}
    fourier_dict = {}
    
    # Loading ASTER
    dict_info_path = os.path.join(path_dataset, 'results/bicubic/{}_dict_pred.pkl'.format(image_index))
    
    with open(dict_info_path, 'rb') as f:
        dict_info = pickle.load(f)
    
    l,c = dict_info['LST_ASTER'].shape[0], dict_info['LST_ASTER'].shape[1]
    temperature_dict["ASTER"] = dict_info['LST_ASTER']
    temperature_dict["NDVI"] = dict_info['NDVI']
    
    if l < 50 or c<50:
        continue
    
    # Making the fourier data.
    aster_tens = torch.tensor(temperature_dict["ASTER"]).unsqueeze(0).unsqueeze(0)
    aster_hf = aster_tens - us.get_output_ftm(aster_tens, mtf = 0.1)

    fourier_dict["ASTER"] = np.fft.fftshift(np.abs(sp.fft.fft2(temperature_dict["ASTER"])))
    fourier_dict["NDVI"] = np.fft.fftshift(np.abs(sp.fft.fft2(temperature_dict["NDVI"])))
    
    # Loading the predictions
    ## This is probably badly optimized as we repeat the same computation multiple times
    for i, model in enumerate(models):
        # Loading LST_SR
        dict_info_path = os.path.join(path_dataset, 'results/', model, "{}_dict_pred.pkl".format(image_index))
        
        with open(dict_info_path, 'rb') as f:
            dict_info = pickle.load(f)

        temperature_dict[model] = dict_info['LST_SR']
        fourier_dict[model] = np.fft.fftshift(np.abs(sp.fft.fft2(temperature_dict[model])))

    # Now that the loading and processing is done, for simplicity we add ASTER inside of the two lists.
    models.insert(0, 'ASTER')
    labels.insert(0, 'ASTER')

    models.insert(1, 'NDVI')
    labels.insert(1, 'NDVI')

    # Getting the 1D attenuation spectra
    attenuation_spectra_dict = {}
    for i,model in enumerate(models):
        attenuation_spectra_dict[model] = us.compute_2D_attenuation_spectra(fourier_dict[model])
   
    list_spectra.append(attenuation_spectra_dict)
    Fmin = 1/(231.65 * (2*len(attenuation_spectra_dict['bicubic']))) 
    Fe = 1/231.65
    x_axis = np.linspace(Fmin, Fe, len(attenuation_spectra_dict['ASTER']))
    list_x_axis.append(x_axis)

# First, we find the x_axis starting with the highest frequency
maxi_size = len(list_x_axis[0])
maxi_idx = 0
for idx, x_axis in enumerate(list_x_axis): 
    if len(x_axis)>maxi_size:
        maxi_size = len(x_axis)
        maxi_idx = idx

# Setting the grid to interpolate to
to_grid = list_x_axis[maxi_idx]

# Doing the interpolation
average_dict_spectra = {}
std_dict_spectra = {}

for i,model in enumerate(models):
    average_dict_spectra[model] = np.zeros(len(to_grid))
    std_dict_spectra[model] = np.zeros(len(to_grid))

for idx in range(len(list_x_axis)):
    
    grid_init = list_x_axis[idx] # We start from here and go to to_grid.

    attenuation_spectra_dict_interp = {}
    for i,model in enumerate(models):
        attenuation_spectra_dict_interp[model] = np.interp(to_grid, grid_init, np.array(list_spectra[idx][model]))
        average_dict_spectra[model] += attenuation_spectra_dict_interp[model]
        std_dict_spectra[model] += np.power(attenuation_spectra_dict_interp[model], 2)
        
# Computing the average spectrum
for i,model in enumerate(models):
    average_dict_spectra[model] /= len(keys_list)
    std_dict_spectra[model] = np.sqrt(std_dict_spectra[model]/len(keys_list) - np.power(average_dict_spectra[model], 2))

# #%%
# # # Store data 
# # with open('./avg_spectrum.pkl', 'wb') as f:
# #     pickle.dump(average_dict_spectra, f)
# # with open('./std_spectrum.pkl', 'wb') as f:
# #     pickle.dump(std_dict_spectra, f)
# # with open('./grid.pkl', 'wb') as f:
# #     pickle.dump(to_grid, f)

# #%% 
# import pickle
# import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,})
# plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}')

# with open('./avg_spectrum.pkl', 'rb') as f:
#     average_dict_spectra = pickle.load(f)
# with open('./std_spectrum.pkl', 'rb') as f:
#     std_dict_spectra = pickle.load(f)
# with open('./grid.pkl', 'rb') as f:
#     to_grid = pickle.load(f)

fig, ax = plt.subplots(1, figsize = (10,7))
fig.subplots_adjust(wspace=0.15, hspace=0.1)
line1, = ax.plot(to_grid, average_dict_spectra['ASTER'], color = 'red', linewidth=3)
line2, = ax.plot(to_grid, average_dict_spectra['bicubic'], color = 'darkviolet')
line3, = ax.plot(to_grid, average_dict_spectra['TsHARP'], color = 'blue', marker = '+', markersize = 3)
line4, = ax.plot(to_grid, average_dict_spectra['ATPRK'], color = 'cyan', marker = '1', markersize = 3)
line5, = ax.plot(to_grid, average_dict_spectra['DMS'], color = 'turquoise', marker = 'x', markersize = 3)
line6, = ax.plot(to_grid, average_dict_spectra['modelB_2011'], color = 'black')
line7, = ax.plot(to_grid, average_dict_spectra['modelB_1009'], color = 'darkred', marker = '+', markersize = 3)
line8, = ax.plot(to_grid, average_dict_spectra['modelB_2609'], color = 'orange', marker = 'x', markersize = 3)
line9, = ax.plot(to_grid, average_dict_spectra['NDVI'], color = 'red',linestyle='dashed')

line1.set_label('ASTER')
line2.set_label('Bicubic')
line3.set_label('TsHARP')
line4.set_label('ATPRK')
line5.set_label('DMS')
line6.set_label('SC-Unet')
line7.set_label('SIF-NN-SR1') 
line8.set_label('SIF-NN-SR2')
line9.set_label('NDVI')

ax.legend(fontsize = 16)
ax.set_xlabel(r'Spatial frequencies $(m^{-1})$', fontsize = 16)
ax.set_ylabel('Attenuation (dB)', fontsize = 16)
ax.tick_params(labelsize=14)
ax.set_xlim(0,0.0043)

# plt.savefig('./figures_test_dataset/average_attenuation_spectrum.pdf',format='pdf',bbox_inches='tight')
# plt.close('all')

#%%!!! Making an archive with all the figures for all the pairs in the test dataset.
# Parameters for the latex text inside the figures
plt.rcParams.update({
    "text.usetex": True,})
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

path_dataset = "./test_data_formatted"
path_figures = "./figures_test_dataset"

os.makedirs(path_figures, exist_ok = True)

# We load the dictionnary of results for bicubic only to extract the indices
df = pd.read_csv(os.path.join(path_dataset, 'results','bicubic', "performances.csv"))
keys_list = np.array(df['Unnamed: 0'].to_numpy()[:-7], dtype = np.int16)

# image_index = keys_list[0]
for image_index in keys_list:
    models = ['bicubic','TsHARP', 'ATPRK', 'DMS', 'modelB_2011', 'modelB_1009', 'modelB_2609']
    labels = ['bicubic','TsHARP', 'ATPRK', 'DMS', 'SC-Unet', 'SIF-NN-SR1', 'SIF-NN-SR2']
    
    temperature_dict = {}
    hf_temp_dict = {}
    
    for i, model in enumerate(models):
        if not i: # For ASTER
            dict_info_path = os.path.join(path_dataset, 'results/bicubic/{}_dict_pred.pkl'.format(image_index))
            with open(dict_info_path, 'rb') as f:
                    dict_info = pickle.load(f)
            
            # Getting the temperatures
            l,c = dict_info['LST_ASTER'].shape[0], dict_info['LST_ASTER'].shape[1]
            temperature_dict['ASTER'] = dict_info['LST_ASTER'].flatten()
            
            # Getting the HF content
            aster_tensor = torch.tensor(dict_info['LST_ASTER']).unsqueeze(0).unsqueeze(0)
            aster_hf = aster_tensor - us.get_output_ftm(aster_tensor)
            hf_temp_dict["ASTER"] = aster_hf.numpy()[0,0,:,:].flatten()
            
        ## For the approaches
        # Getting the temperature
        dict_info_path = os.path.join(path_dataset, 'results', model,'{}_dict_pred.pkl'.format(image_index))
        with open(dict_info_path, 'rb') as f:
                dict_info = pickle.load(f)
        temperature_dict[model] = dict_info["LST_SR"].flatten()
        
        # Getting the HF content
        aster_tensor = torch.tensor(dict_info["LST_SR"]).unsqueeze(0).unsqueeze(0)
        aster_hf = aster_tensor - us.get_output_ftm(aster_tensor)
        hf_temp_dict[model] = aster_hf.numpy()[0,0,:,:].flatten()
    
    # Loading the LST and the NDVI
    dict_info_path = os.path.join(path_dataset, 'results/bicubic/{}_dict_pred.pkl'.format(image_index))
    with open(dict_info_path, 'rb') as f:
            dict_info = pickle.load(f)
    lst_modis = dict_info['LST:'] #!!! Yes there is a slight mistake in the key here
    ndvi = dict_info['NDVI']
    
    # Limitating the dynamic range to ASTER's as the other predictions may have a problem 
    # in their ranges.
    mini = np.min(temperature_dict["ASTER"])
    maxi = np.max(temperature_dict["ASTER"])
    
    ## 1st figure
    fig = plt.figure(1, figsize = (10,10))
    gs = gridspec.GridSpec(6, 6, fig)
    fig.subplots_adjust(wspace=0.003, hspace=0.3)
    
    ax1 = plt.subplot(gs[:2, :2])
    ig1 = ax1.imshow(lst_modis, cmap = 'jet', vmin = mini, vmax = maxi) 
    ax1.axis('off')
    ax1.set_title("LST")
    
    ax2 = plt.subplot(gs[:2, 2:4])
    ig2 = ax2.imshow(temperature_dict["ASTER"].reshape(l,c), cmap = 'jet', vmin = mini, vmax = maxi) 
    ax2.axis('off')
    ax2.set_title(r"$LST_{ASTER}$")
    
    ax3 = plt.subplot(gs[:2, 4:])
    ig3 = ax3.imshow(temperature_dict["bicubic"].reshape(l,c), cmap = 'jet', vmin = mini, vmax = maxi) 
    ax3.axis('off')
    ax3.set_title(r"$LST_{SR}$"+" Bic")
    
    ax4 = plt.subplot(gs[2:4, :2])
    ig4 = ax4.imshow(temperature_dict["TsHARP"].reshape(l,c), cmap = 'jet', vmin = mini, vmax = maxi) 
    ax4.axis('off')
    ax4.set_title(r"$LST_{SR}$"+" TsHARP")
    
    ax5 = plt.subplot(gs[2:4, 2:4])
    ig5 = ax5.imshow(temperature_dict["ATPRK"].reshape(l,c), cmap = 'jet', vmin = mini, vmax = maxi) 
    ax5.axis('off')
    ax5.set_title(r"$LST_{SR}$"+" ATPRK")
    
    ax6 = plt.subplot(gs[2:4, 4:])
    ig6 = ax6.imshow(temperature_dict["DMS"].reshape(l,c), cmap = 'jet', vmin = mini, vmax = maxi) 
    ax6.axis('off')
    ax6.set_title(r"$LST_{SR}$"+" DMS")
    
    ax9 = plt.subplot(gs[4:, :2])
    ig9 = ax9.imshow(temperature_dict["modelB_2011"].reshape(l,c), cmap = 'jet', vmin = mini, vmax = maxi) 
    ax9.axis('off')
    ax9.set_title(r"$LST_{SR}$"+" SC-Unet ")
    
    ax8 = plt.subplot(gs[4:, 2:4])
    ig8 = ax8.imshow(temperature_dict["modelB_1009"].reshape(l,c), cmap = 'jet', vmin = mini, vmax = maxi) 
    ax8.axis('off')
    ax8.set_title(r"$LST_{SR}$"+" SIF-NN-SR1")
    
    ax9 = plt.subplot(gs[4:, 4:])
    ig9 = ax9.imshow(temperature_dict["modelB_2609"].reshape(l,c), cmap = 'jet', vmin = mini, vmax = maxi) 
    ax9.axis('off')
    ax9.set_title(r"$LST_{SR}$"+" SIF-NN-SR2")
    
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7]) # left, bottom, width, height
    fig.colorbar(ig1, cax = cbar_ax)
    plt.show()
    
    plt.savefig(os.path.join(path_figures,'{}_prediction_comparison_tds.pdf'.format(image_index)),format='pdf',bbox_inches='tight')
    
    ## 2nd figure
    labels.insert(0, 'ASTER')
    
    fig, ax = plt.subplots(1,2, figsize = (23,7))
    fig.subplots_adjust(wspace=0.15, hspace=0.1)
    # Temperature predictions
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].boxplot(temperature_dict.values())
    ax[0].violinplot(temperature_dict.values())
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel('Temperature T (K)', fontsize = 14)
    
    # High-freq content
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].boxplot(hf_temp_dict.values())
    ax[1].violinplot(hf_temp_dict.values())
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel(r'T - K $\ast$ I (K)', fontsize = 14)
    plt.savefig(os.path.join(path_figures,'{}_prediction_distribution_tds.pdf'.format(image_index)),format='pdf',bbox_inches='tight')
    
    labels = labels[1:]
    
    ## 3rd Figure
    # In this part, we don't look at the BICUBIC because as it "doesnt possess"
    # HF content and it doesnt really downsample the data.
    temperature_dict = {}
    reg_dict = {}
    
    dict_info_path = os.path.join(path_dataset, 'results/bicubic/{}_dict_pred.pkl'.format(image_index))
    with open(dict_info_path, 'rb') as f:
            dict_info = pickle.load(f)
    
    # Getting the temperatures
    l,c = dict_info['LST_ASTER'].shape[0], dict_info['LST_ASTER'].shape[1]
    temperature_dict["ASTER"] = dict_info['LST_ASTER']
    
    # Loading the predictions
    for i, model in enumerate(models):
        dict_info_path = os.path.join(path_dataset, 'results', model,'{}_dict_pred.pkl'.format(image_index))
        with open(dict_info_path, 'rb') as f:
                dict_info = pickle.load(f)
        temperature_dict[model] = dict_info['LST_SR']
        
        reg_dict[model] = sp.stats.linregress(temperature_dict["ASTER"].flatten(), temperature_dict[model].flatten())
        
    min_aster = np.min(temperature_dict["ASTER"].flatten())
    max_aster = np.max(temperature_dict["ASTER"].flatten())
    
    min_predicted = np.min([temperature_dict['TsHARP'].flatten(), temperature_dict["ATPRK"].flatten(), temperature_dict["DMS"].flatten(), temperature_dict["modelB_1009"].flatten(),temperature_dict["modelB_2609"].flatten(), temperature_dict["modelB_2011"].flatten()])
    max_predicted = np.max([temperature_dict['TsHARP'].flatten(), temperature_dict["ATPRK"].flatten(), temperature_dict["DMS"].flatten(), temperature_dict["modelB_1009"].flatten(),temperature_dict["modelB_2609"].flatten(), temperature_dict["modelB_2011"].flatten()])
    
    fig, ax = plt.subplots(2,3, figsize = (16,11))
    for i, model in enumerate(models[1:]):
        x = i // 3
        y = i % 3
        reg = reg_dict[model]
        
        img = ax[x][y].hexbin(temperature_dict["ASTER"].flatten(), temperature_dict[model].flatten(), gridsize=50,  bins = 'log')
        plt.colorbar(img, ax = ax[x][y])
        ax[x][y].plot([min_aster, max_aster],[reg.slope * min_aster + reg.intercept, reg.slope * max_aster + reg.intercept], color = 'red')
        ax[x][y].grid()
        if reg.intercept > 0:
            ax[x][y].set_title(labels[i+1]+": " + r" $LST_{SR}$" +" = {0:.2g}".format(reg.slope) + r" $LST_{ASTER}$"+"+ {0:.2g}".format(reg.intercept)+", R² = {0:.2g}".format(reg.rvalue**2))
        else:
            ax[x][y].set_title(labels[i+1]+": " + r" $LST_{SR}$" +" = {0:.2g}".format(reg.slope) + r" $LST_{ASTER}$"+"{0:.2g}".format(reg.intercept)+", R² = {0:.2g}".format(reg.rvalue**2))
        ax[x][y].set_ylabel("MODIS Downscaled Temperature (K)", fontsize = 14)
        ax[x][y].set_xlabel("ASTER Temperature (K)", fontsize = 14)
        ax[x][y].set_xlim([min_aster, max_aster])
        ax[x][y].set_ylim([min_predicted-1, max_predicted+1])
    
    fig.subplots_adjust(right=1.2)
    plt.savefig(os.path.join(path_figures,'{}_prediction_temperature_tds.pdf'.format(image_index)),format='pdf',bbox_inches='tight')
    
    # 4th figure
    # Choosing the approaches
    temperature_dict = {}
    fourier_dict = {}
    
    # Loading ASTER
    dict_info_path = os.path.join(path_dataset, 'results/bicubic/{}_dict_pred.pkl'.format(image_index))
    with open(dict_info_path, 'rb') as f:
        dict_info = pickle.load(f)
    
    l,c = dict_info['LST_ASTER'].shape[0], dict_info['LST_ASTER'].shape[1]
    temperature_dict["ASTER"] = dict_info['LST_ASTER']
    temperature_dict["NDVI"] = dict_info['NDVI']
    
    # Making the fourier data. 
    # For ASTER
    aster_tens = torch.tensor(temperature_dict["ASTER"]).unsqueeze(0).unsqueeze(0)
    fourier_dict["ASTER"] = np.fft.fftshift(np.abs(sp.fft.fft2(temperature_dict["ASTER"])))
    
    # For the NDVI
    ndvi_tens = torch.tensor(temperature_dict["NDVI"]).unsqueeze(0).unsqueeze(0)
    fourier_dict["NDVI"] = np.fft.fftshift(np.abs(sp.fft.fft2(temperature_dict["NDVI"])))
    
    # For the predictions
    for i, model in enumerate(models):
        # Loading LST_SR
        dict_info_path = os.path.join(path_dataset, 'results/', model, "{}_dict_pred.pkl".format(image_index))
        
        with open(dict_info_path, 'rb') as f:
            dict_info = pickle.load(f)
    
        temperature_dict[model] = dict_info['LST_SR']
        fourier_dict[model] = np.fft.fftshift(np.abs(sp.fft.fft2(temperature_dict[model])))
    
    # Now that the loading and processing is done, for simplicity we add ASTER inside of the two lists.
    models.insert(0, 'ASTER')
    labels.insert(0, 'ASTER')
    
    models.insert(1, 'NDVI')
    labels.insert(1, 'NDVI')
    
    # Getting the 1D attenuation spectra
    attenuation_spectra_dict = {}
    for i,model in enumerate(models):
        attenuation_spectra_dict[model] = us.compute_2D_attenuation_spectra(fourier_dict[model])
    
    Fmin = 1/(231.65 * (2*len(attenuation_spectra_dict['bicubic']))) 
    Fe = 1/231.65
    x_axis = np.linspace(Fmin, Fe, len(attenuation_spectra_dict['ASTER']))
    
    fig, ax = plt.subplots(1, figsize = (10,7))
    fig.subplots_adjust(wspace=0.15, hspace=0.1)
    line1, = ax.plot(x_axis, attenuation_spectra_dict['ASTER'], color = 'red', linewidth=3)
    line2, = ax.plot(x_axis, attenuation_spectra_dict['bicubic'], color = 'darkviolet')
    line3, = ax.plot(x_axis, attenuation_spectra_dict['TsHARP'], color = 'blue', marker = '+', markersize = 3)
    line4, = ax.plot(x_axis, attenuation_spectra_dict['ATPRK'], color = 'cyan', marker = '1', markersize = 3)
    line5, = ax.plot(x_axis, attenuation_spectra_dict['DMS'], color = 'turquoise', marker = 'x', markersize = 3)
    line9, = ax.plot(x_axis, attenuation_spectra_dict['modelB_2011'], color = 'black')
    line7, = ax.plot(x_axis, attenuation_spectra_dict['modelB_2609'], color = 'orange', marker = 'x', markersize = 3)
    line6, = ax.plot(x_axis, attenuation_spectra_dict['modelB_1009'], color = 'darkred', marker = '+', markersize = 3)
    line8, = ax.plot(x_axis, attenuation_spectra_dict['NDVI'], color = 'red',linestyle='dashed')
    
    line1.set_label('ASTER')
    line2.set_label('Bicubic')
    line3.set_label('TsHARP')
    line4.set_label('ATPRK')
    line5.set_label('DMS')
    line9.set_label('SC-Unet')
    line6.set_label('SIF-NN-SR1') 
    line7.set_label('SIF-NN-SR2')
    line8.set_label('NDVI')
    
    ax.legend(fontsize = 16)
    ax.set_xlabel(r'Spatial frequencies $(m^{-1})$', fontsize = 16)
    ax.set_ylabel('Attenuation (dB)', fontsize = 16)
    ax.tick_params(labelsize=14)
    ax.set_xlim(0,0.0043)
    
    plt.savefig(os.path.join(path_figures,'{}_prediction_frequencies_single_tds.pdf'.format(image_index)),format='pdf',bbox_inches='tight')
    plt.close('all')

#%% Observing the contrast inversion problem for SIF-NN-SR1 and for DMS
path_predictions = './test_data2_formatted/results'

image_index = 0
path_DMS = os.path.join(path_predictions,'DMS', '{}_dict_pred.pkl'.format(image_index))
path_SIFNNSR1 = os.path.join(path_predictions,'modelB_1009', '{}_dict_pred.pkl'.format(image_index))

with open(path_DMS, 'rb') as f:
    dict_info = pickle.load(f)
    lst_aster = dict_info['LST_ASTER']
    lst_dms = dict_info['LST_SR']
with open(path_SIFNNSR1, 'rb') as f:
    dict_info = pickle.load(f)
    lst_cnn1 = dict_info['LST_SR']

mini = np.min([lst_aster])
maxi = np.max([lst_aster])

fig, ax = plt.subplots(1,3, figsize = (25,9))
fig.subplots_adjust(wspace=0.05, hspace=0.095)

ig1 = ax[0].imshow(lst_aster, cmap = 'jet', vmin = mini, vmax = maxi) 
ax[0].axis('off')
ax[0].set_title(r"$LST_{ASTER}$", fontsize = 25)

ig2 = ax[1].imshow(lst_cnn1, cmap = 'jet', vmin = mini, vmax = maxi) 
ax[1].axis('off')
ax[1].set_title(r"$LST_{SR, CNN1}$", fontsize = 25)

ig3 = ax[2].imshow(lst_dms, cmap = 'jet', vmin = mini, vmax = maxi) 
ax[2].axis('off')
ax[2].set_title(r"$LST_{SR, DMS}$", fontsize = 25)

cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7]) # left, bottom, width, height
fig.colorbar(ig1, cax = cbar_ax)
# plt.savefig('./figures/contrast_inversion_tds2_0.pdf',format='pdf',bbox_inches='tight')
