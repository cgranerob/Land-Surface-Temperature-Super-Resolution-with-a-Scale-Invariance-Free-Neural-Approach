#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Utilitary file containing the definition of most of the functions used. It also
contains the code for TsHARP, ATPRK, and for computing the gaussian kernel / high 
frequency content. 

Else
----
@author: Romuald Ait Bachir
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

import subprocess
from json import load as json_load
import copy 
import os
import pickle as pkl

import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from scipy.ndimage import sobel
from scipy.stats import linregress
import scipy.optimize as opt
from scipy.spatial.distance import pdist, squareform

from osgeo import gdal

#%%
###############################################################################
def date_into_n_chunk(l_days, n):
    """
    Description
    -----------
    Splitting the date list into n part in order to download the MODIS data 
    in a parallelized fashion. 
    
    """
    lst = [i for i in range(len(l_days))]
    size = ceil(len(lst) / n)
    l = list(map(lambda x: lst[x * size:x * size + size], list(range(n))))
    return [[l_days[k[0]],l_days[k[-1]]] for k in l]

###############################################################################
# Used in process_modis.py
def compute_NDVI(nir, red):
    """
    Function computing the Normalized Difference Vegetation Index.
    Comments: Something could maybe be done in order for it to not care about the clouds.
        
    Parameters
    ----------
    nir : array of float64
        Near Infrared Image of the ground.
    red : array of float64
        Visible (in the red) image of the ground..

    Returns
    -------
    ndvi : array of float64
        NDVI image of the ground

    """
    return (nir - red)/(nir + red)

def process_LST_QC(QC):
    """This function segments the good pixels from the bad ones in the Land Surface Temperature"""
    QC[np.bitwise_and(QC, 0b11)<2] = 1
    QC[np.bitwise_and(QC, 0b11)>=2] = 0
    return QC

def split(LST, QC, window_size):
    """Generates the processed LST QC and LST crops"""
        
    for i in range(0, LST.shape[0], window_size[0]):
        for j in range(0, LST.shape[1], window_size[1]):
            yield (j,i,LST[j:j+window_size[1], i:i+window_size[0]],QC[j:j+window_size[1], i:i+window_size[0]])
 
def split_NIRRed(NIR, Red ,window_size):
    """
    The breaking point of this function will be realized inside the main function
    Parameters
    ----------
    NIRRed : TYPE
        Either Nir or Red.
    window_size : TYPE
        DESCRIPTION.
    cnt : TYPE
        DESCRIPTION.

    Yields
    ------
    TYPE
        DESCRIPTION.
    """
    for i in range(0, NIR.shape[0], window_size[0]):
        for j in range(0, NIR.shape[1], window_size[1]):
            yield (j,i,NIR[j:j+window_size[1], i:i+window_size[0]],Red[j:j+window_size[1], i:i+window_size[0]])

###############################################################################
# Finding the corners in the MODIS and ASTER Reprojected patches into UTM.
# Used in model_perf_aster_formatds.py
def condition_lst_modis(image):
    image[image<200] = 0
    image[image>=200] = 1
    return image

def condition_lst_aster(image):
    # The threshold is 10x since the scaling factor is still not applied.
    image[image<230] = 0
    image[image>=230] = 1
    return image

def find_corners(image, condition):
    img_quadr = image.copy()
    img_quadr = condition(img_quadr)

    sobel_h = sobel(img_quadr, 0)  # horizontal gradient
    sobel_v = sobel(img_quadr, 1)  # vertical gradient
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    magnitude *= 255.0 / np.max(magnitude)  # normalization
    
    img_quadr = img_quadr.astype('uint8')
    img_quadr = cv2.cvtColor(255*img_quadr, cv2.COLOR_GRAY2BGR)
    
    # convert image to gray scale image 
    img_quadr = cv2.cvtColor(img_quadr, cv2.COLOR_BGR2GRAY) 
      
    # detect corners with the goodFeaturesToTrack function. 
    edges = cv2.goodFeaturesToTrack(img_quadr, 4, 0.01, 10) 
    
    if edges is None:
        return None, None
    else:
        edges = np.intp(edges) 
        if edges.shape != (4,1,2): # This shouldnt be a problem in absolute...
            return None, None
        else:
            edges = edges.reshape(4,2)
            edges = np.array([list(p) for p in edges])
            # mini1 = np.min(edges[:,0])
            # maxi1 = np.max(edges[:,0])
            # mini2 = np.min(edges[:,1])
            # maxi2 = np.max(edges[:,1])

            e0 = np.sort(edges[:,0])
            p1 = [e0[1], e0[2]]
            
            e1 = np.sort(edges[:,1])
            p2 = [e1[1], e1[2]]
    
    return p1, p2

###############################################################################
# Upsampling and downsampling the images.
def upsampling(img, scale):
    """
    Bicubic interpolation. Used when making the dataset.

    Parameters
    ----------
    img : numpy.array
        Image to interpolate.
    scale : tuple or list
        Increasing resolution's factor.

    Returns
    -------
    image_out : numpy array
        Interpolated image.
        
    """
    return cv2.resize(img, dsize = (img.shape[0]*scale[0], img.shape[1]*scale[1]), fx = scale[0], fy = scale[1], interpolation=cv2.INTER_CUBIC)


def downsampling(img, scale = (4,4)):
    """
    Description
    -----------
    Faster implementation of the Norm L4 downsampling.
    
    The unfold function is not so easy to understand. Check: 
    https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
    for a better understanding.
    
    Parameters
    ----------
    img : tensor
        Tensor containing generated super resolved lst images. Shape: (Batch_size, 1, 256, 256)
    scale : tuple or list
        Decreasing resolution's factor.

    Returns
    -------
    img : tensor
        Downsampled image.

    """
    # Unfolding the image batch into blocks of dimension 4x4
    img = img.unfold(dimension = 3, size = scale[0], step = scale[0])
    img = img.unfold(dimension = 2, size = scale[1], step = scale[1])
    
    # Computing the distance
    img = torch.pow(img, 4)
    img = torch.sum(img, dim = (-1,-2)) / (scale[0]*scale[1])
    return torch.pow(img, 0.25)


def downsampling_img(img, scale):
    """

    Parameters
    ----------
    img : numpy.array
        Array containing generated super resolved lst images. Shape: (256, 256)
    scale : tuple or list
        Decreasing resolution's factor.

    Returns
    -------
    img : tensor
        Downsampled image.

    """
    
    img_out = np.zeros((img.shape[0]//scale[0], img.shape[1]//scale[1]))
    for i in range(0, img.shape[0],scale[0]):
        for j in range(0, img.shape[1], scale[1]):
            val = np.power(np.sum(np.power(img[i:i+scale[0],j:j+scale[1]],4)) / (scale[0]*scale[1]), 1/4)
            img_out[i//scale[0],j//scale[1]] = val
    return img_out


###############################################################################
# Reading the .hdf/.tif data and metadata.
def get_dataset(file, idx_subdataset):
    info = subprocess.check_output(["gdalinfo", file]).decode('ascii').split("\n")
    
    # Take the subdatasets names list from the information.
    info = info[info.index("Subdatasets:")+1:info.index("Corner Coordinates:")]
    
    # Extracting the name of our subdatasets found at its index. 
    subds_name = info[2*idx_subdataset].split('=')[1]
    
    # Now extracting the subdataset's informations.
    info = subprocess.check_output(["gdalinfo", subds_name]).decode('ascii').split("\n")
    
    return info

def get_geotransform(file, idx_subdataset):
    # First, we run the command line, decode the binary output into a string 
    # and split it in order to have one info per line.
    info = subprocess.check_output(["gdalinfo", file]).decode('ascii').split("\n")
    
    # Take the subdatasets names list from the information.
    info = info[info.index("Subdatasets:")+1:info.index("Corner Coordinates:")]
    
    # Extracting the name of our subdatasets found at its index. 
    subds_name = info[2*idx_subdataset].split('=')[1]
    
    # Now extracting the subdataset's informations.
    info = subprocess.check_output(["gdalinfo", subds_name]).decode('ascii').split("\n")
    
    # Finding the georeference informations.
    for i in range(len(info)):
        if "Origin =" in info[i]:
            break
    
    # Getting the elements, extracting the tuples (string format) and evaluating them.
    origin = eval(info[i].split(' ')[-1])
    precision = eval(info[i+1].split(' ')[-1])
    
    geotransform = [origin[0], precision[0], 0, origin[1], 0, precision[1]]
    
    return geotransform

def read_LST(file : str,
             time : str = 'day'):
    """
    Description
    -----------
    Function reading the important information inside the .hdf files from the 
    MOD11A1 product.
    
    PROBLEM: IT SEEMS LIKE THE GetGeotransforms from gdal doesn't really work 
    on my computer. GT[0], GT[3] and GT[5] are all zeros while they shouldn't.
    
    In order to correct it, the geotransform is extracted by directly using gdalinfo
    in a function entitled get_geotransform.
    
    Parameters
    ----------
    file : str
        Path to the LST .hdf file to read.
    time : str, optional
        "Day" or "Night" to choose the right layer. The default is 'day'.

    Returns
    -------
    LST_K : np.array
        Day or Night LST in kelvin.
    LST_QC : np.array
        Quality control image of the LST.
    cols : int
        X dimension of the image.
    rows : int
        Y dimension of the image.
    projection : str
        Projection information of the data.
    geotransform : list
        Geotransform parameters obtained from gdalinfo.

    """
    if time == 'day':
        idx_day = 0
        # open dataset Day
        dataset = gdal.Open(file,gdal.GA_ReadOnly)
        subdataset = gdal.Open(dataset.GetSubDatasets()[idx_day][0], gdal.GA_ReadOnly)
    
        cols =subdataset.RasterXSize
        rows = subdataset.RasterYSize
        projection = subdataset.GetProjection()
        # geotransform = subdataset.GetGeoTransform() #Affine transform of the coefficients
        geotransform = get_geotransform(file, idx_day)
        
        # We read the Image as an array
        band = subdataset.GetRasterBand(1)
        LST_raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float32)
    
        # To convert LST MODIS units to Kelvin
        LST_K_day=0.02*LST_raw
    
        subdataset = gdal.Open(dataset.GetSubDatasets()[1][0], gdal.GA_ReadOnly)
    
        # We read the Image as an array
        band = subdataset.GetRasterBand(1)
        LST_QC = band.ReadAsArray(0, 0, cols, rows).astype(np.uint8)
    
        return LST_K_day, LST_QC, cols, rows, projection, geotransform
    
    elif time == 'night':
        # open dataset Night
        idx_night = 4
        dataset = gdal.Open(file,gdal.GA_ReadOnly)
        subdataset = gdal.Open(dataset.GetSubDatasets()[idx_night][0], gdal.GA_ReadOnly)
    
        cols =subdataset.RasterXSize
        rows = subdataset.RasterYSize
        projection = subdataset.GetProjection()
        # geotransform = subdataset.GetGeoTransform()
        geotransform = get_geotransform(file, idx_night)
        
        # We read the Image as an array
        band = subdataset.GetRasterBand(1)
        LST_raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float32)
    
        # To convert LST MODIS units to Kelvin
        LST_K_night=0.02*LST_raw
    
        subdataset = gdal.Open(dataset.GetSubDatasets()[5][0], gdal.GA_ReadOnly)
    
        # We read the Image as an array
        band = subdataset.GetRasterBand(1)
        LST_QC = band.ReadAsArray(0, 0, cols, rows).astype(np.uint8)
    
        return LST_K_night, LST_QC, cols, rows, projection, geotransform


def read_NIRRED(file: str):
    """
    Description
    -----------
    Function reading the important information inside the .hdf files from the 
    MOD09GQ product.
    
    PROBLEM: IT SEEMS LIKE THE GetGeotransforms from gdal doesn't really work 
    on my computer. GT[0], GT[3] and GT[5] are all zeros while they shouldn't.
    
    In order to correct it, the geotransform is extracted by directly using gdalinfo
    in a function entitled get_geotransform.
    
    Parameters
    ----------
    file : str
        Path to the NIR_Red .hdf file to read.

    Returns
    -------
    Red : np.array
        Red image from MOD09GQ.
    NIR : np.array
        Near Infrared image from MOD09GQ.
    cols : int
        X dimension of the image.
    rows : int
        Y dimension of the image.
    projection : str
        Projection information of the data.
    geotransform : list
        Geotransform parameters obtained from gdalinfo.

    """
    dataset = gdal.Open(file,gdal.GA_ReadOnly)
    subdataset = gdal.Open(dataset.GetSubDatasets()[1][0], gdal.GA_ReadOnly)

    cols =subdataset.RasterXSize
    rows = subdataset.RasterYSize
    projection = subdataset.GetProjection()
    # geotransform = subdataset.GetGeoTransform()
    geotransform = get_geotransform(file, 1)

	# Coordinates of top left pixel of the image (Lat, Lon)
	# coords=np.asarray((geotransform[0],geotransform[3]))

	# We read the Image as an array
    band = subdataset.GetRasterBand(1)
    raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float32)
    # bandtype = gdal.GetDataTypeName(band.DataType)

    # To convert LST MODIS units to Kelvin
    Red=0.0001*raw
    subdataset = gdal.Open(dataset.GetSubDatasets()[2][0], gdal.GA_ReadOnly)

	# We read the Image as an array
    band = subdataset.GetRasterBand(1)
    raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float32)
    
    NIR=0.0001*raw
    
    return Red, NIR, cols, rows, projection, geotransform

def read_LST_aster(file : str):
    """Only used to read the array"""
    idx_day = 0
    # open dataset Day
    dataset = gdal.Open(file,gdal.GA_ReadOnly)
    subdataset = gdal.Open(dataset.GetSubDatasets()[idx_day][0], gdal.GA_ReadOnly)

    cols =subdataset.RasterXSize
    rows = subdataset.RasterYSize
    projection = subdataset.GetProjection()
    geotransform = subdataset.GetGeoTransform() #Affine transform of the coefficients
    
    # We read the Image as an array
    band = subdataset.GetRasterBand(1)
    LST_raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float32)

    # To convert LST ASTER units to Kelvin
    LST_K_day=0.1*LST_raw

    subdataset = gdal.Open(dataset.GetSubDatasets()[1][0], gdal.GA_ReadOnly)

    # We read the Image as an array
    band = subdataset.GetRasterBand(1)

    return LST_K_day, cols, rows, projection, geotransform

def read_MOD44W(file: str):
    """
    Description
    -----------

    Parameters
    ----------
    file : str
        Path to the NIR_Red .hdf file to read.

    Returns
    -------
    water_mask
    water_qc
    cols : int
        X dimension of the image.
    rows : int
        Y dimension of the image.
    projection : str
        Projection information of the data.
    geotransform : list
        Geotransform parameters obtained from gdalinfo.

    """
    dataset = gdal.Open(file,gdal.GA_ReadOnly)
    subdataset = gdal.Open(dataset.GetSubDatasets()[0][0], gdal.GA_ReadOnly)

    cols =subdataset.RasterXSize
    rows = subdataset.RasterYSize
    projection = subdataset.GetProjection()
    # geotransform = subdataset.GetGeoTransform()
    geotransform = get_geotransform(file, 1)

	# Coordinates of top left pixel of the image (Lat, Lon)
	# coords=np.asarray((geotransform[0],geotransform[3]))

	# We read the Image as an array
    band = subdataset.GetRasterBand(1)
    water = band.ReadAsArray(0, 0, cols, rows).astype(np.uint8)

    return water, cols, rows, projection, geotransform


def read_GeoTiff(file):
    # open dataset Day
	dataset = gdal.Open(file,gdal.GA_ReadOnly)
	#subdataset =	gdal.Open(dataset.GetSubDatasets()[0][0], gdal.GA_ReadOnly)

	cols =dataset.RasterXSize
	rows = dataset.RasterYSize
	projection = dataset.GetProjection()
	geotransform = dataset.GetGeoTransform() # Not problematic since it should have been transformed

	# Coordinates of top left pixel of the image (Lat, Lon)
	# coords=np.asarray((geotransform[0],geotransform[3]))

	# We read the Image as an array
	band = dataset.GetRasterBand(1)
	image = band.ReadAsArray(0, 0, cols, rows).astype(np.float32)

	return image, cols, rows, projection, geotransform


def save_GeoTiff(cropped_img, out_file, projection, geotransform):
    """
    The preprocessing is realized outside of this function. Modified from CCC"""
    
    img = np.zeros((1,cropped_img.shape[0],cropped_img.shape[1]))
    img[0,:,:] = cropped_img
    
    driver = gdal.GetDriverByName("GTiff")
    outDs = driver.Create(out_file, img.shape[1], img.shape[2], 1, gdal.GDT_Float32) 
    outDs.SetProjection(projection)
    outDs.SetGeoTransform(geotransform) 

    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(img[0,:,:])
    outDs.FlushCache()
    return True

###############################################################################
# Metrics evaluated during the training and evaluation of the models:

def psnr_skimage(predictions, targets):
    psnr = []
    for i in range(targets.shape[0]):
        psnr.append(peak_signal_noise_ratio(targets[i,0,:,:], predictions[i,0,:,:], data_range=targets.max() - targets.min()))
    return np.mean(psnr)

def ssim_skimage(predictions, targets):
    """
    Description
    -----------
    Use this one for now!!

    Parameters
    ----------
    predictions : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    ssim = []
    for i in range(targets.shape[0]):
        targ = targets[i,0,:,:]
        pred = predictions[i,0,:,:]
        ssim.append(structural_similarity(targ, pred, data_range=targets.max() - targets.min()))
    return np.mean(ssim)

###############################################################################
# ????? Didn't check to see if it is used somewhere...
def linear_regression_from_sklearn(linear_regression):
    ### Is this even used ???
    bias = linear_regression.intercept_
    coef = linear_regression.coef_
    class affine:
        def __init__(self, a, b):
            self.a = a[0,0]
            self.b = b[0]
        def predict(self, x):
            return self.a*x + self.b
        
    return affine(coef, bias)

###############################################################################
# For the computation of the 1D attenuation spectra and the metrics in the 
# fourier space
def compute_2D_attenuation_spectra(im):
    # Im refers to the 2D fourier transform of the image.
    # Equation of a circle: (x-c0)²  + (y-c1)² = r²
    center = (im.shape[0]//2, im.shape[1]//2)

    # Generating the matrix containing the positions.
    pos_matrix = np.zeros((2,im.shape[0], im.shape[1]))

    for i in range(im.shape[0]):
        pos_matrix[0, i, :] = i

    for j in range(im.shape[1]):
        pos_matrix[1, :, j] = j

    # Getting the distance and allowing only a certain quantity of error
    r_range = 1
   
    # Getting the value at f0
    intensity_f0 = im[center[0], center[1]]
    
    # Getting the other values
    attenuation_spectrum = [intensity_f0/intensity_f0]
    for r in range(0, min([center[0] - r_range, center[1] - r_range])):
        # Getting the circles
        small_circle = r**2 - ((pos_matrix[0] - center[0])**2 + (pos_matrix[1] - center[1])**2)
        small_circle[small_circle>=0] = 1
        small_circle[small_circle<0] = 0
        
        big_circle = (r+r_range)**2 - ((pos_matrix[0] - center[0])**2 + (pos_matrix[1] - center[1])**2)
        big_circle[big_circle>=0] = 1
        big_circle[big_circle<0] = 0
        
        # Getting the mask
        frequency_mask = big_circle - small_circle
        
        # Getting the intensity in the fourier space
        spectrum = np.sum(im*frequency_mask)/np.sum(frequency_mask)
        attenuation_spectrum.append(10 * (np.log10(np.array(spectrum)) - np.log10(np.array(intensity_f0))))
        
    return attenuation_spectrum

def get_PFR(rb, xb):
    # rb refers to ground-truth and xb to the bicubic
    return np.sum([max([rb[i] - xb[i], 0]) for i in range(len(rb))])

def get_AFR(pb, rb, xb):
    # pb refers to prediction, rb refers to ground-truth and xb to the bicubic
    tot_sum = 0
    for i in range(len(pb)):
        t1 = min([pb[i], rb[i]])
        t2 = min([xb[i], rb[i]])
        t3 = min([rb[i], xb[i]])
        tot_sum += max([t1, t2]) - t3
    return tot_sum

def get_FRR(pb, rb, xb):
    return get_AFR(pb, rb, xb) / get_PFR(rb, xb)

def get_FRO(pb, rb, xb):
    den = np.sum(rb)
    return (den ** -1) * np.sum([ rb[i] - max([pb[i], rb[i]]) for i in range(len(pb))])

def get_FRU(pb, rb, xb):
    den = np.sum(xb)
    return (den ** -1) * np.sum([ xb[i] - min([pb[i], xb[i]]) for i in range(len(pb))])


###############################################################################
# Checkpointing the model
class model_checkpoint:
    # Basically like the Early Stopping in keras
    def __init__(self, n_epochs, patience = 5):
        self.patience = patience
        self.curr_patience = 0
        self.saved_state = None
        self.saved_best_value = None
        self.curr_epoch = None
        self.best_epoch = None
        self.max_epochs = n_epochs
        self.train_state = None
        
    def test_update(self, model, metrics, val_monitored, epoch):
        self.curr_epoch = epoch
        
        if self.curr_epoch == 1:
            self.best_epoch = self.curr_epoch
            self.saved_state = copy.deepcopy(model.state_dict())
            self.saved_best_value = metrics[val_monitored][-1]
            
        else:
            if metrics[val_monitored][-1] >= self.saved_best_value:
                self.curr_patience += 1
                
                # First possibility: The patience counter gets bigger than the max patience authorized.
                # Exiting the training
                if self.curr_patience >= self.patience:
                    self.train_state = 'break'
                    
                # Second possibility: The patience counter is not 0 when the max epoch happens
                elif self.curr_patience > 0 and self.curr_epoch == self.max_epochs:
                    self.train_state = 'break'
                    
                else:
                    self.train_state = 'continue'
                    
            else:
                self.best_epoch = self.curr_epoch
                self.curr_patience = 0
                self.saved_best_value = metrics[val_monitored][-1]
                self.saved_state = copy.deepcopy(model.state_dict())
                
                # Third possibility: Max epoch happens and patience is 0
                if self.curr_epoch == self.max_epochs:
                    self.train_state = 'continue'
                
                else:
                    self.train_state = 'continue'

###############################################################################
# Reading the training hyperparameters 
def read_JsonA(file):
    """
    Description
    -----------
    Function reading a JSON file containing all the training parameters for the modelA.
    TODO Remake description
    
    Parameters
    ----------
    file : str
        Path to the paramsX.json file
    """
    
    data = json_load(open(file))
    
    dataset_parameter = data['dataset_parameter']
    device = data['device']
    hyperparameters = data['hyperparameters']
    modelA_parameters = data['modelA_parameters']
    save_parameters = data['save_parameters']
    return dataset_parameter, modelA_parameters, hyperparameters, save_parameters, device


def read_JsonB(file):
    """
    Description
    -----------
    Function reading a JSON file containing all the training parameters for the modelB.
    It also includes the parameters from the ModelA since we need those to run the 
    training of ModelB.
    
    Note: The modelA parameters are the remnants of another approach not demonstrated.
    
    Parameters
    ----------
    file : str
        Path to the paramsX.json file
    """
    
    data = json_load(open(file))
    dataset_parameter = data['dataset_parameter']
    device = data['device']
    hyperparameters = data['hyperparameters']
    modelA_parameters = data['modelA_parameters']
    modelB_parameters = data['modelB_parameters']
    save_parameters = data['save_parameters']
    return dataset_parameter, modelA_parameters, modelB_parameters, hyperparameters, save_parameters, device


def read_JsonC(file):
    """
    Description
    -----------
    Function reading a JSON file containing all the training parameters for the modelC.
    
    NOT USED.
    
    Parameters
    ----------
    file : str
        Path to the paramsX.json file
    """
    
    data = json_load(open(file))
    dataset_parameter = data['dataset_parameter']
    device = data['device']
    hyperparameters = data['hyperparameters']
    modelC_parameters = data['modelC_parameters']
    save_parameters = data['save_parameters']
    return dataset_parameter, modelC_parameters, hyperparameters, save_parameters, device

###############################################################################
# Loading/Saving the model
def load_model(model, state_dict_file, device = 'cpu'):
    """
    example of file:
    './results/modelA_bil/modelA_state_dict.pt'
    """
    if device == 'cpu':
        model.load_state_dict(torch.load(state_dict_file, map_location = torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(state_dict_file))
        

def save_model(model, path, model_name):
    """
    Description
    -----------
    Saving the model. For redundancy, the model is saved two times.
    1. By saving its state_dict
    2. By saving its whole definition, weights, ... The whole variable.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model to save.
    path : string
        Path where to save the model
    model_name : string
        Name of the model.

    """
    #1.State_dict
    sd_name = os.path.join(path, model_name+'_'+'state_dict.pt')
    torch.save(model.state_dict(), sd_name)
    
    #2. Model
    md_name = os.path.join(path, model_name+'.pt')
    torch.save(model, md_name)

###############################################################################
# Reading the stored losses
def read_losses(path):
    """
    Description
    -----------
    Function plotting the losses stored into the pickle file.

    Parameters
    ----------
    path: string
        path to the model folder.
    """
    with open(os.path.join(path, "modelB_lossdata.pkl"), 'rb') as f:
        metrics_dict = pkl.load(f)
        
    return metrics_dict

###############################################################################
# ===========================================================================
# Thunmpy code modified in order to change the prototypes to directly work with 
# numpy arrays.

# Should see the github to have all the interesting comments.
# =============================================================================

def linear_fit(index, Temp, min_T):
    # Note: Here, index refers to the NDVI at coarse resolution which is obtained
    #bu using the L4 downsampling on the reflectance.
    T=Temp.flatten()
    
    T_great_mask=np.greater(T,min_T) # We eliminate the background pixels and those with a temperature under min_T
    T=T[T_great_mask]
    
    I=index.flatten() 
    I=I[T_great_mask]
    	
	# We look for nans in the index and we eliminate those pixels from the index and the temperature
    NaN=np.isfinite(I)
    I=I[NaN]
    T=T[NaN]
    	
    # linear regression
    fit=linregress(I,T)

    print('Fit Done')
    return fit

def linear_unmixing(index, Temp, fit_param, iscale, mask=0):
    (rows,cols)=index.shape
    
    ## Param[1] indicates that we call the intercept of the linear regression.
    a0 = fit_param[1]
    ## Param[0] indicates that we call the slope of the linear regression.
    a1 = fit_param[0]

    T_unm = a0 + a1*index
    
    if mask == 0:
        maskt=cv2.resize(Temp, (T_unm.shape[1],T_unm.shape[0]), interpolation = cv2.INTER_NEAREST)
        maskt[maskt!=0]=1
        T_unm=T_unm*maskt
    else:
        zz=np.where(mask==0) 
        T_unm[zz]=0
    
    print('Unmixing Done')
    return T_unm

def correction_linreg(index, Temp, TT_unm, iscale, fit_param):

    (rows,cols)=TT_unm.shape
    
    ## Param[1] indicates that we call the intercept of the linear regression.
    a0 = fit_param[1]
    ## Param[0] indicates that we call the slope of the linear regression.
    a1 = fit_param[0]
    
    # We define the matrix of model temperature	
    T_unm_add = a0 + a1*index
            
    mask=np.greater(Temp,0) # We eliminate the background pixels (not in the image)
    T_unm_add[~mask]=0
    
    #  We define the delta at the scale of the measured temperature as in the model.
    Delta_T = Temp - T_unm_add
    
    # We define a mask in order to not use the background pixels. If a pixel of the finer scale image is zero, 
    # the coarse pixel which contains this one is also zero. We have to choose this mask or the mask of line 188
    mask_TT_unm=np.zeros((rows,cols))
    mask_TT_unm[np.nonzero(TT_unm)]=1
    
    # We obtain the delta at the scale of the unmixed temperature
    Delta_T_final =np.zeros((rows,cols))
    for ic in range(int(np.floor(cols/iscale))):
        for ir in range(int(np.floor(rows/iscale))):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if mask_TT_unm[ir_2,ic_2]==0:
                        Delta_T_final[ir_2,ic_2] = 0
                    else:
                        Delta_T_final[ir_2,ic_2] = Delta_T[ir,ic] 
    
    # We correct the unmixed temperature using the delta	
    TT_unmixed_corrected = TT_unm + Delta_T_final
    
    print('Correction Done')

    return TT_unmixed_corrected


def Func_Gamma_cc(pd_uni_c, sill, ran):	
    Gamma_c_temp = sill * (1 - np.exp(-pd_uni_c/(ran/3))) 	# EXPONENTIAL	
    return Gamma_c_temp


def Gamma_ff(pd_uni_c, sill, ran, dis_f, N_c, iscale, pd_c):
    
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_cc=np.zeros((N_c,N_c))
    
    for i_coarse in range(N_c):
        for j_coarse in range(N_c):
            temp_var=0
            for i_fine in range(iscale*iscale):
                for i in range(iscale*iscale):
                    Gamma_ff_temp[i] = sill * (1 - np.exp(-dis_f[i_coarse,j_coarse,i_fine,i]/(ran/3))) 	# EXPONENTIAL
                temp_var = sum(Gamma_ff_temp) + temp_var
            Gamma_cc[i_coarse,j_coarse] = 1/(iscale**4) * temp_var

    # We need to classify the Gamma_cc values of pixels i and j in function of the distance between the coarse 
    # i pixel and the coarse j pixel.

    Gamma_cc_m=np.zeros(len(pd_uni_c))
    
    for idist in range(len(pd_uni_c)):
        ii=0
        for i_coarse in range(N_c):
            for j_coarse in range(N_c):
                if pd_c[i_coarse,j_coarse] == pd_uni_c[idist]:
                    ii=ii+1
                    Gamma_cc_m[idist] = Gamma_cc_m[idist] + Gamma_cc[i_coarse,j_coarse]
            Gamma_cc_m[idist] = Gamma_cc_m[idist]/ii	

    Gamma_cc_r=Gamma_cc_m-Gamma_cc_m[0]
    

    return Gamma_cc_r


def correction_ATPRK(index, Temp, TT_unm, fit_param, iscale, scc, block_size, sill, ran, path_plot=False):

    b_radius = int(np.floor(block_size/2))

    (rows,cols)=TT_unm.shape
    (rows_t,cols_t)=Temp.shape
################# CORRECTION #######################################

    ## Param[1] indicates that we call the intercept of the linear regression.
    a0 = fit_param[1]
    ## Param[0] indicates that we call the slope of the linear regression.
    a1 = fit_param[0]
    
    # We define the matrix of model temperature	
    T_unm_add = a0 + a1*index
            
    mask=np.greater(Temp,0) # We eliminate the background pixels (not in the image)
    T_unm_add[~mask]=0
    
    #  We define the delta at the scale of the measured temperature as in the model.
    Delta_T = Temp - T_unm_add

    #####################################################################
    ################ ATPK METHOD ########################################
    #####################################################################
    			
    #####################################################################
    ######### 1 Coarse semivariogram estimation #########################
    #####################################################################

    # We define the matrix of distances
    dist=np.zeros((2*block_size-1,2*block_size-1))
    for i in range(-(block_size-1),block_size):
        for j in range(-(block_size-1),block_size):
            dist[i+(block_size-1),j+(block_size-1)]=scc*np.sqrt(i**2+j**2)

    # We define the vector of different distances
    distances= np.unique(dist)

    new_matrix=np.zeros(((block_size)**2,3))

    Gamma_coarse_temp=np.zeros((rows_t,cols_t,len(distances)))

    for irow in range(b_radius,rows_t-b_radius):
        for icol in range(b_radius,cols_t-b_radius):
            dt = Delta_T[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]	
            for ir in range(block_size):
                for ic in range(block_size):
                    new_matrix[ir*(block_size) + ic,0] = scc*ir
                    new_matrix[ir*(block_size) + ic,1] = scc*ic
                    new_matrix[ir*(block_size) + ic,2] = dt[ir,ic]
            pd_c = squareform(pdist( new_matrix[:,:2] ))
            pd_uni_c = np.unique(pd_c)
            N_c = pd_c.shape[0]
            for idist in range(len(pd_uni_c)):
                idd=pd_uni_c[idist]
                if idd==0:
                    Gamma_coarse_temp[irow,icol,idist]=0
                else:
                    ii=0	
                    for i in range(N_c):
                        for j in range(i+1,N_c):
                            if pd_c[i,j] == idd:
                                ii=ii+1
                                Gamma_coarse_temp[irow,icol,idist] = Gamma_coarse_temp[irow,icol,idist] + ( new_matrix[i,2] - new_matrix[j,2] )**2.0
                    Gamma_coarse_temp[irow,icol,idist]	= Gamma_coarse_temp[irow,icol,idist]/(2*ii)		
		
    Gamma_coarse=np.zeros(len(distances))
    for idist in range(len(pd_uni_c)):
        zz=np.nonzero(Gamma_coarse_temp[:,:,idist])
        gg=Gamma_coarse_temp[zz[0],zz[1],idist]
        Gamma_coarse[idist]=np.mean(gg)
	
    Gamma_coarse[np.isnan(Gamma_coarse)]=0	
	
    sillran_c=np.array([sill,ran])
	
    ydata=Gamma_coarse
    xdata=pd_uni_c
	
    param_c, pcov_c = opt.curve_fit(Func_Gamma_cc, xdata, ydata,sillran_c,method='lm')
    
    if path_plot!=False:
        plt.plot(distances,param_c[0] * (1 - np.exp(-distances/(param_c[1]/3))))
        plt.plot(distances,Gamma_coarse)
        plt.savefig(path_plot)
        plt.close()
	
    #####################################################################
    ######### 2 Deconvolution ###########################################
    #####################################################################
    	
    dist_matrix=np.zeros(((iscale*block_size)**2,2))
	
    for ir in range(iscale*block_size):
        for ic in range(iscale*block_size):
            dist_matrix[ir*(iscale*block_size) + ic,0] = scc/iscale * ir
            dist_matrix[ir*(iscale*block_size) + ic,1] = scc/iscale * ic	
	
    pd_f = squareform( pdist( dist_matrix[:,:2] ) )
    #pd_uni_f = np.unique(pd_f)
    #N_f = pd_f.shape[0]
	
    dis_f=np.zeros((N_c,N_c,iscale*iscale,iscale,iscale))
	
    for ii in range(block_size): # We select a coarse pixel
        for jj in range(block_size): # We select a coarse pixel
            #temp_variable=[]
            for iiii in range(iscale): # We take all the fine pixels inside the chosen coarse pixel
                count=0
                for counter in range(jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale,jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale+iscale):	
                    res=np.reshape(pd_f[counter,:],(block_size*iscale,block_size*iscale))
                    for icoarse in range(block_size):
                        for jcoarse in range(block_size):
                            dis_f[ii*(block_size)+jj,icoarse*(block_size) + jcoarse,iiii*iscale+count,:,:] = res[icoarse*iscale:icoarse*iscale+iscale,jcoarse*iscale:jcoarse*iscale+iscale]
                    count=count+1		
    
    #Comment: dis_f is a matrix of distances. The first dimension select the coarse i pixel, the second dimension select the coarse j pixel, the third dimension, 
    #         select the fine pixel inside the i coarse pixel and the two last dimensions give us the distance from the fine pixel (third dimension) of i to all 
    #         the fine pixels of j.
	
    ####### We reshape dis_f to convert the last two dimensions iscale, iscale into one dimension iscale*iscale
    dis_f= np.reshape(dis_f,(N_c,N_c,iscale*iscale,iscale*iscale))
		
    sill=param_c[0]
    ran=param_c[1]
    
    sillran=np.array([sill,ran])
	
    ydata=Gamma_coarse
    xdata=pd_uni_c
	
    param, pcov= opt.curve_fit(lambda pd_uni_c, sill, ran: Gamma_ff(pd_uni_c, sill, ran, dis_f, N_c, iscale, pd_c), xdata, ydata, sillran, method='lm') 

	
    #####################################################################
    ######### 3 Estimation Gamma_cc and Gamma_fc ########################
    #####################################################################
	
    ####################### Gamma_cc estimation ######################### 
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_cc=np.zeros((N_c,N_c))
		
    for i_coarse in range(N_c):
        for j_coarse in range(N_c):
            temp_var=0
            for i_fine in range(iscale*iscale):
                for i in range(iscale*iscale):
                    Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[i_coarse,j_coarse,i_fine,i]/(param[1]/3))) 	# EXPONENTIAL
                temp_var = sum(Gamma_ff_temp) + temp_var	
            Gamma_cc[i_coarse,j_coarse] =  temp_var/(iscale**4)
    
    # We need to classify the Gamma_cc values of pixels i and j in function of the distance between the coarse 
    # i pixel and the coarse j pixel.
    	
    # Gamma_cc_regularized estimation and comparison with Gamma_coarse	
    Gamma_cc_m=np.zeros(len(pd_uni_c))
    	
    for idist in range(len(pd_uni_c)):
        ii=0
        for i_coarse in range(N_c):
            for j_coarse in range(N_c): 	
                if pd_c[i_coarse,j_coarse] == pd_uni_c[idist]:
                    ii=ii+1
                    Gamma_cc_m[idist] = Gamma_cc_m[idist] + Gamma_cc[i_coarse,j_coarse]
        Gamma_cc_m[idist] = Gamma_cc_m[idist]/ii	
	
    # Gamma_cc_r=Gamma_cc_m-Gamma_cc_m[0]
	
    # gamma_cc_regul compared to gamma_coarse	
    # Diff = Gamma_coarse - Gamma_cc_r
	
    ####################### Gamma_fc estimation #########################
	
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_fc=np.zeros((iscale*iscale,N_c))
	
    for j_coarse in range(N_c):
        temp_var=0
        for i_fine in range(iscale*iscale):
            for i in range(iscale*iscale):
                Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[int(np.floor(0.5*block_size**2)),j_coarse,i_fine,i]/(param[1]/3))) 	# EXPONENTIAL
            temp_var = sum(Gamma_ff_temp) 
            Gamma_fc[i_fine,j_coarse] =  temp_var/(iscale**2)
							
    #####################################################################
    ######### 4 Weight estimation (lambdas) #############################
    #####################################################################	
    	
    vec_right = np.ones((block_size**2,1))
    vec_bottom = np.append(np.ones(block_size**2),0)
    	
    A = np.vstack((np.hstack((Gamma_cc,vec_right)),vec_bottom))
    
    Ainv = np.linalg.inv(A)
    	
    B=np.zeros((iscale*iscale,N_c+1))
    lambdas_temp=np.zeros((iscale*iscale,N_c+1))
    for i_fine in range(iscale*iscale):
        B[i_fine,:] = np.append(Gamma_fc[i_fine,:],1) 
        lambdas_temp[i_fine,:] = np.dot(Ainv,B[i_fine,:])	
	
    # lambdas first dimension is the fine pixel inside the central coarse pixel in a (block_size x block_size) pixels block. 
    # lambdas second dimension is the coars pixels in this block.	
    lambdas =lambdas_temp[:,0:len(lambdas_temp[0])-1]
	
    #####################################################################
    ######### 5 Fine Residuals estimation ###############################
    #####################################################################			
	
    mask_TT_unm=np.zeros((rows,cols))
    indice_mask_TT=np.nonzero(TT_unm)
    mask_TT_unm[indice_mask_TT]=1
	
    # We obtain the delta at the scale of the unmixed temperature
    Delta_T_final =np.zeros((rows,cols))
	
    for ic in range(b_radius,int(np.floor(cols/iscale)-b_radius)):
        for ir in range(b_radius,int(np.floor(rows/iscale)-b_radius)):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if mask_TT_unm[ir_2,ic_2]==0:
                        Delta_T_final[ir_2,ic_2] = 0
                    else:
                        temp_var = np.reshape(Delta_T[ir-b_radius:ir+b_radius+1,ic-b_radius:ic+b_radius+1],block_size**2) # We take the block of coarse pixels
                        # We multiply each coarse pixel in the block by the corresponding lambda and sum
                        Delta_T_final[ir_2,ic_2] = np.sum(lambdas[(ir_2-ir*iscale)*iscale+ic_2-ic*iscale,:]*temp_var)
	
    # We correct the unmixed temperature using the delta	
    TT_unmixed_corrected = TT_unm + Delta_T_final
    
    print('Correction Done')
	
    return TT_unmixed_corrected

def TsHARP(temp_coarse, index_coarse, index_fine, scale, min_T=285, path_image=False):

    fit = linear_fit(index_coarse, temp_coarse, min_T)
    temp_fine = linear_unmixing(index_fine, temp_coarse, fit, scale, mask=0)
    temp_fine_corrected = correction_linreg(index_coarse, temp_coarse, temp_fine, scale, fit)
    
    # if path_image!=False:
    #     driver = gdal.GetDriverByName("ENVI")
    #     outDs = driver.Create(path_image, int(cols), int(rows), 1, gdal.GDT_Float32)
    #     outDs.SetProjection(projection_h)
    #     outDs.SetGeoTransform(geotransform_h)
    #     outBand = outDs.GetRasterBand(1)
    #     outBand.WriteArray(T_H_corrected, 0, 0)
    #     outBand.FlushCache()
    #     outDs = None

    print('TsHARP Done')
    
    return temp_fine_corrected


def ATPRK(temp_coarse, index_coarse, index_fine, scale, scc, block_size=5, sill=7, ran=1000, min_T=285, path_image=False):


    fit = linear_fit(index_coarse, temp_coarse, min_T)
    temp_fine = linear_unmixing(index_fine, temp_coarse, fit, scale, mask=0)
    temp_fine_corrected = correction_ATPRK(index_coarse, temp_coarse, temp_fine, fit, scale, scc, block_size, sill, ran, path_plot=False)
    
    # if path_image!=False:
    #     driver = gdal.GetDriverByName("ENVI")
    #     outDs = driver.Create(path_image, int(cols), int(rows), 1, gdal.GDT_Float32)
    #     outDs.SetProjection(projection_h)
    #     outDs.SetGeoTransform(geotransform_h)
    #     outBand = outDs.GetRasterBand(1)
    #     outBand.WriteArray(T_H_corrected, 0, 0)
    #     outBand.FlushCache()
    #     outDs = None

    print('ATPRK Done')
    
    return temp_fine_corrected


def linear_fit_window(index, Temp, min_T, b_radius):
########### PROCESSED VERSION OF T AND I AND LINEAR REGRESSION #######
    
    (rows,cols)=Temp.shape

    T=Temp.flatten()

    T_great_mask=np.greater(T,min_T) # We eliminate the background pixels and those with a temperature under min_T
    T=T[T_great_mask]

    I=index.flatten() 
    I=I[T_great_mask]
	
    # We look for nans in the index and we eliminate those pixels from the index and the temperature
    NaN=np.isfinite(I)
    I=I[NaN]
    T=T[NaN]
	
# Two different linear regressions

    fit2=linregress(I,T)
	
#####################################################################
################### WINDOW DEFINITION , #############################
########### AND LINEAR REGRESSION FOR EACH WINDOW ###################

    a1 = np.zeros((rows, cols))
    a0 = np.zeros((rows, cols))

    for irow in range(b_radius,rows-b_radius):
        print(irow, end=" ")
        for icol in range(b_radius,cols-b_radius):
            
            Tw = Temp[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
            Iw = index[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
            Tw = Tw.flatten()
            Iw = Iw.flatten()
            mask=np.greater(Tw,min_T) # We eliminate the background pixels and those with a temperature under min_T K
            Tw=Tw[mask]
            Iw=Iw[mask]
            NaN=np.isfinite(Iw) # We look for nans in the index and we eliminate those pixels from the index and the temperature
            Iw=Iw[NaN]
            Tw=Tw[NaN]
            if len(Tw) > 2/3*(2*b_radius+1)**2:
                fit=linregress(Iw,Tw)
                a1[irow,icol]=fit[0]
                a0[irow,icol]=fit[1]
            if len(Tw) <= 2/3*(2*b_radius+1)**2:
                a1[irow,icol]=fit2[0]
                a0[irow,icol]=fit2[1]
	
    for irow in range(0,b_radius):
        for icol in range(cols):			
            a1[irow,icol]=fit2[0]
            a0[irow,icol]=fit2[1]
    
    for irow in range(rows-b_radius,rows):
        for icol in range(cols):			
            a1[irow,icol]=fit2[0]
            a0[irow,icol]=fit2[1]
    			
    for icol in range(0,b_radius):
        for irow in range(rows):			
            a1[irow,icol]=fit2[0]
            a0[irow,icol]=fit2[1]
			
    for icol in range(cols-b_radius,cols):
        for irow in range(rows):			
            a1[irow,icol]=fit2[0]
            a0[irow,icol]=fit2[1]
    

    print('Fit Done')
	
    return a0, a1


def aatprk_unmixing(index, Temp, Slope_Intercept, iscale):

    (rows,cols)=index.shape
    
    I_mask=np.greater(np.absolute(index),0.0000000000) # We eliminate the background pixels (not in the image)
    T_unm=np.zeros((rows,cols))
    for ic in range(int(np.floor(cols/iscale))):
        for ir in range(int(np.floor(rows/iscale))):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if I_mask[ir_2,ic_2] == False:
                        T_unm[ir_2,ic_2]=0
                    else:
                        T_unm[ir_2,ic_2] = Slope_Intercept[1,ir,ic] + Slope_Intercept[0,ir,ic] * index[ir_2,ic_2]
    

    print('Unmixing Done')
	
    return T_unm

def correction_AATPRK(index, Temp, TT_unm, Slope_Intercept, iscale, scc, block_size, sill, ran, path_plot=False):

    ####################################################################
    ################ LOADING DATA ######################################
    ####################################################################

    b_radius = int(np.floor(block_size/2))
	
    (rows,cols)=TT_unm.shape
    (rows_t,cols_t)=Temp.shape
    ################# CORRECTION #######################################
    
    # We define the matrix of model temperature	
    T_unm_add = Slope_Intercept[1] + Slope_Intercept[0]*index
            
    mask=np.greater(Temp,0) # We eliminate the background pixels (not in the image)
    T_unm_add[~mask]=0
    
    #  We define the delta at the scale of the measured temperature as in the model.															
    Delta_T = Temp - T_unm_add			
	
    #####################################################################
    ################ ATPK METHOD ########################################
    #####################################################################
    			
    #####################################################################
    ######### 1 Coarse semivariogram estimation #########################
    #####################################################################
	
	# We define the matrix of distances
    dist=np.zeros((2*block_size-1,2*block_size-1))
    for i in range(-(block_size-1),block_size):
        for j in range(-(block_size-1),block_size):
            dist[i+(block_size-1),j+(block_size-1)]=scc*np.sqrt(i**2+j**2)
	
    # We define the vector of different distances
    distances= np.unique(dist)		
	
    new_matrix=np.zeros(((block_size)**2,3))
	
    Gamma_coarse_temp=np.zeros((rows_t,cols_t,len(distances)))
	
    for irow in range(b_radius,rows_t-b_radius):
        for icol in range(b_radius,cols_t-b_radius):
            dt = Delta_T[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]	
            for ir in range(block_size):
                for ic in range(block_size):
                    new_matrix[ir*(block_size) + ic,0] = scc*ir
                    new_matrix[ir*(block_size) + ic,1] = scc*ic
                    new_matrix[ir*(block_size) + ic,2] = dt[ir,ic]	
            pd_c = squareform( pdist( new_matrix[:,:2] ) )
            pd_uni_c = np.unique(pd_c)
            N_c = pd_c.shape[0]
            for idist in range(len(pd_uni_c)):
                idd=pd_uni_c[idist]
                if idd==0:
                    Gamma_coarse_temp[irow,icol,idist]=0
                else:
                    ii=0	
                    for i in range(N_c):
                        for j in range(i+1,N_c):
                            if pd_c[i,j] == idd:
                                ii=ii+1
                                Gamma_coarse_temp[irow,icol,idist] = Gamma_coarse_temp[irow,icol,idist] + ( new_matrix[i,2] - new_matrix[j,2] )**2.0
                    Gamma_coarse_temp[irow,icol,idist]	= Gamma_coarse_temp[irow,icol,idist]/(2*ii)		
		
    Gamma_coarse=np.zeros(len(distances))
    for idist in range(len(pd_uni_c)):
        zz=np.nonzero(Gamma_coarse_temp[:,:,idist])
        gg=Gamma_coarse_temp[zz[0],zz[1],idist]
        Gamma_coarse[idist]=np.mean(gg)
	
    Gamma_coarse[np.isnan(Gamma_coarse)]=0	
	
    sillran_c=np.array([sill,ran])
	
    ydata=Gamma_coarse
    xdata=pd_uni_c
	
    param_c, pcov_c = opt.curve_fit(Func_Gamma_cc, xdata, ydata,sillran_c,method='lm')
    
    if path_plot!=False:
        plt.plot(distances,param_c[0] * (1 - np.exp(-distances/(param_c[1]/3))))
        plt.plot(distances,Gamma_coarse)
        plt.savefig(path_plot)
        plt.close()
	
    #####################################################################
    ######### 2 Deconvolution ###########################################
    #####################################################################
    	
    dist_matrix=np.zeros(((iscale*block_size)**2,2))
	
    for ir in range(iscale*block_size):
        for ic in range(iscale*block_size):
            dist_matrix[ir*(iscale*block_size) + ic,0] = scc/iscale * ir
            dist_matrix[ir*(iscale*block_size) + ic,1] = scc/iscale * ic	
	
    pd_f = squareform( pdist( dist_matrix[:,:2] ) )
    #pd_uni_f = np.unique(pd_f)
    #N_f = pd_f.shape[0]
	
    dis_f=np.zeros((N_c,N_c,iscale*iscale,iscale,iscale))
	
    for ii in range(block_size): # We select a coarse pixel
        for jj in range(block_size): # We select a coarse pixel
            #temp_variable=[]
            for iiii in range(iscale): # We take all the fine pixels inside the chosen coarse pixel
                count=0
                for counter in range(jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale,jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale+iscale):	
                    res=np.reshape(pd_f[counter,:],(block_size*iscale,block_size*iscale))
                    for icoarse in range(block_size):
                        for jcoarse in range(block_size):
                            dis_f[ii*(block_size)+jj,icoarse*(block_size) + jcoarse,iiii*iscale+count,:,:] = res[icoarse*iscale:icoarse*iscale+iscale,jcoarse*iscale:jcoarse*iscale+iscale]
                    count=count+1		
    
    #Comment: dis_f is a matrix of distances. The first dimension select the coarse i pixel, the second dimension select the coarse j pixel, the third dimension, 
    #         select the fine pixel inside the i coarse pixel and the two last dimensions give us the distance from the fine pixel (third dimension) of i to all 
    #         the fine pixels of j.
	
    ####### We reshape dis_f to convert the last two dimensions iscale, iscale into one dimension iscale*iscale
    dis_f= np.reshape(dis_f,(N_c,N_c,iscale*iscale,iscale*iscale))
		
    sill=param_c[0]
    ran=param_c[1]
    
    sillran=np.array([sill,ran])
	
    ydata=Gamma_coarse
    xdata=pd_uni_c
	
    param, pcov= opt.curve_fit(lambda pd_uni_c, sill, ran: Gamma_ff(pd_uni_c, sill, ran, dis_f, N_c, iscale, pd_c), xdata, ydata, sillran, method='lm') 

	
    #####################################################################
    ######### 3 Estimation Gamma_cc and Gamma_fc ########################
    #####################################################################
	
    ####################### Gamma_cc estimation ######################### 
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_cc=np.zeros((N_c,N_c))
		
    for i_coarse in range(N_c):
        for j_coarse in range(N_c):
            temp_var=0
            for i_fine in range(iscale*iscale):
                for i in range(iscale*iscale):
                    Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[i_coarse,j_coarse,i_fine,i]/(param[1]/3))) 	# EXPONENTIAL
                temp_var = sum(Gamma_ff_temp) + temp_var	
            Gamma_cc[i_coarse,j_coarse] =  temp_var/(iscale**4)
    
    # We need to classify the Gamma_cc values of pixels i and j in function of the distance between the coarse 
    # i pixel and the coarse j pixel.
    	
    # Gamma_cc_regularized estimation and comparison with Gamma_coarse	
    Gamma_cc_m=np.zeros(len(pd_uni_c))
    	
    for idist in range(len(pd_uni_c)):
        ii=0
        for i_coarse in range(N_c):
            for j_coarse in range(N_c): 	
                if pd_c[i_coarse,j_coarse] == pd_uni_c[idist]:
                    ii=ii+1
                    Gamma_cc_m[idist] = Gamma_cc_m[idist] + Gamma_cc[i_coarse,j_coarse]
        Gamma_cc_m[idist] = Gamma_cc_m[idist]/ii	
	
    # Gamma_cc_r=Gamma_cc_m-Gamma_cc_m[0]
	
    # gamma_cc_regul compared to gamma_coarse	
    # Diff = Gamma_coarse - Gamma_cc_r
	
    ####################### Gamma_fc estimation #########################
	
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_fc=np.zeros((iscale*iscale,N_c))
	
    for j_coarse in range(N_c):
        temp_var=0
        for i_fine in range(iscale*iscale):
            for i in range(iscale*iscale):
                Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[int(np.floor(0.5*block_size**2)),j_coarse,i_fine,i]/(param[1]/3))) 	# EXPONENTIAL
            temp_var = sum(Gamma_ff_temp) 
            Gamma_fc[i_fine,j_coarse] =  temp_var/(iscale**2)
							
    #####################################################################
    ######### 4 Weight estimation (lambdas) #############################
    #####################################################################	
    	
    vec_right = np.ones((block_size**2,1))
    vec_bottom = np.append(np.ones(block_size**2),0)
    	
    A = np.vstack((np.hstack((Gamma_cc,vec_right)),vec_bottom))
    
    Ainv = np.linalg.inv(A)
    	
    B=np.zeros((iscale*iscale,N_c+1))
    lambdas_temp=np.zeros((iscale*iscale,N_c+1))
    for i_fine in range(iscale*iscale):
        B[i_fine,:] = np.append(Gamma_fc[i_fine,:],1) 
        lambdas_temp[i_fine,:] = np.dot(Ainv,B[i_fine,:])	
	
    # lambdas first dimension is the fine pixel inside the central coarse pixel in a (block_size x block_size) pixels block. 
    # lambdas second dimension is the coars pixels in this block.	
    lambdas =lambdas_temp[:,0:len(lambdas_temp[0])-1]
	
    #####################################################################
    ######### 5 Fine Residuals estimation ###############################
    #####################################################################			
	
    mask_TT_unm=np.zeros((rows,cols))
    indice_mask_TT=np.nonzero(TT_unm)
    mask_TT_unm[indice_mask_TT]=1
	
    # We obtain the delta at the scale of the unmixed temperature
    Delta_T_final =np.zeros((rows,cols))
	
    for ic in range(b_radius,int(np.floor(cols/iscale)-b_radius)):
        for ir in range(b_radius,int(np.floor(rows/iscale)-b_radius)):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if mask_TT_unm[ir_2,ic_2]==0:
                        Delta_T_final[ir_2,ic_2] = 0
                    else:
                        temp_var = np.reshape(Delta_T[ir-b_radius:ir+b_radius+1,ic-b_radius:ic+b_radius+1],block_size**2) # We take the block of coarse pixels
                        # We multiply each coarse pixel in the block by the corresponding lambda and sum
                        Delta_T_final[ir_2,ic_2] = np.sum(lambdas[(ir_2-ir*iscale)*iscale+ic_2-ic*iscale,:]*temp_var)
	
    # We correct the unmixed temperature using the delta	
    TT_unmixed_corrected = TT_unm + Delta_T_final
    
    print('Correction Done')
	
    return TT_unmixed_corrected


def AATPRK(temp_coarse, index_coarse, index_fine, scale, scc, b_radius=2, block_size=5, sill=7, ran=1000, min_T=285, path_image=False):
    
    Intercept, Slope = linear_fit_window(index_coarse, temp_coarse, min_T, b_radius)
    temp_fine = aatprk_unmixing(index_fine, temp_coarse, np.asarray([Slope,Intercept]), scale)
    temp_fine_corrected = correction_AATPRK(index_coarse, temp_coarse, temp_fine, np.asarray([Slope,Intercept]), scale, scc, block_size, sill, ran, path_plot=False)
    
    # if path_image!=False:
    #     driver = gdal.GetDriverByName("ENVI")
    #     outDs = driver.Create(path_image, int(cols), int(rows), 1, gdal.GDT_Float32)
    #     outDs.SetProjection(projection_h)
    #     outDs.SetGeoTransform(geotransform_h)
    #     outBand = outDs.GetRasterBand(1)
    #     outBand.WriteArray(T_H_corrected, 0, 0)
    #     outBand.FlushCache()
    #     outDs = None

    print('AATPRK Done')
    
    return temp_fine_corrected

###############################################################################
# LST fine scale to coarse scale using a PSF. Starter Code given by Julien Michel.
# Slight modifications made. Not all the functions are probably used. Some functions 
# were added but the syntax and efficiency weren't really searched.

import math

def generate_psf_kernel(
    res: float, mtf_res: float, mtf_fc: float, half_kernel_width: None = None
) -> np.ndarray:
    """
    Generate a psf convolution kernel
    """
    fc = 0.5 / mtf_res 
    sigma = math.sqrt(-math.log(mtf_fc) / 2) / (math.pi * fc)
    if half_kernel_width is None:
        half_kernel_width = int(math.ceil(mtf_res / (res)))
    kernel = np.zeros((2 * half_kernel_width + 1, 2 * half_kernel_width + 1))
    for i in range(0, half_kernel_width + 1):
        for j in range(0, half_kernel_width + 1):
            dist = res * math.sqrt(i**2 + j**2)
            psf = np.exp(-(dist * dist) / (2 * sigma * sigma)) / (
                sigma * math.sqrt(2 * math.pi)
            )
            kernel[half_kernel_width - i, half_kernel_width - j] = psf
            kernel[half_kernel_width - i, half_kernel_width + j] = psf
            kernel[half_kernel_width + i, half_kernel_width + j] = psf
            kernel[half_kernel_width + i, half_kernel_width - j] = psf

    kernel = kernel / np.sum(kernel)
    kernel = kernel.astype(np.float32)
    return kernel

# This function could be refactored in order to be applied directly into the training when computing the reconstruction.
def generic_downscale(
    data: torch.Tensor,
    factor: float = 2.0,
    mtf: float = 0.1,
    padding="same",
    hkw: int = 3,
):
    """
    Downsample patches with proper aliasing filtering
    """
    # Generate psf kernel for target MTF
    psf_kernel = torch.tensor(
        generate_psf_kernel(1.0, factor, mtf, hkw), device=data.device, dtype=data.dtype
    )
    # Convolve data with psf kernel
    data = torch.nn.functional.pad(data, (hkw, hkw, hkw, hkw), mode="reflect")
    data = torch.nn.functional.conv2d(
        data,
        psf_kernel[None, None, :, :].expand(data.shape[1], -1, -1, -1),
        groups=data.shape[1],
        padding=padding,
    )
    # Downsample with nearest neighbors
    data = torch.nn.functional.interpolate(
        data, scale_factor=1 / factor, mode="bicubic"
    )
    return data


def downscale_LST_SR_to_LR(
    data: np.ndarray,
    factor: float = 4,
    mtf: float = 0.1,
    padding="same",
    hkw: None = None,
    deci_type = 'bic'
):
    """
    Downsample patches with proper aliasing filtering
    """
    # Generate psf kernel for target MTF
    psf_kernel = torch.tensor(
        generate_psf_kernel(1.0, factor, mtf, hkw), device=data.device, dtype=data.dtype
    )
    
    half_width = int((psf_kernel.shape[-1] - 1) / 2)
    
    # Convolve data with psf kernel
    data = torch.nn.functional.pad(data, (half_width, half_width, half_width, half_width), mode="reflect")
    
    data = torch.nn.functional.conv2d(
        data,
        psf_kernel[None, None, :, :].expand(data.shape[1], -1, -1, -1),
        groups=data.shape[1],
        padding=padding,
    )
    if deci_type == 'bic':
        # Downsample with nearest neighbors
        data = torch.nn.functional.interpolate(
            data, scale_factor=1 / factor, mode="bicubic"
        )
        
        size_loss = int(half_width/factor)
        
        return data[: ,: ,size_loss:data.shape[-2] - size_loss, size_loss:data.shape[-1] - size_loss]
    
    elif deci_type == 'norm-L4':
        
        data = data[: ,: ,half_width:data.shape[-2] - half_width, half_width:data.shape[-1] - half_width]
        
        data = downsampling(data, scale = (4,4))
        
        return data

def downscale_LST_SR_to_LR_test(
    data: np.ndarray,
    factor: float = 4,
    mtf: float = 0.1,
    padding="same",
    hkw: None = None,
    deci_type = 'bic'
):
    """
    Downsample patches with proper aliasing filtering
    """
    
    data_tensor = torch.Tensor(data).unsqueeze(0).unsqueeze(0)
    
    # Generate psf kernel for target MTF
    psf_kernel = torch.tensor(
        generate_psf_kernel(1.0, factor, mtf, hkw), device = data_tensor.device, dtype=data_tensor.dtype
    )
    
    half_width = int((psf_kernel.shape[-1] - 1) / 2)
    
    # Convolve data with psf kernel
    data = torch.nn.functional.pad(data_tensor, (half_width, half_width, half_width, half_width), mode="reflect")
    
    if deci_type == 'bic':
        # Downsample with nearest neighbors
        data = torch.nn.functional.interpolate(
            data, scale_factor=1 / factor, mode="bicubic"
        )
        
        size_loss = int(half_width/factor)
        
        return data[: ,: ,size_loss:data.shape[-2] - size_loss, size_loss:data.shape[-1] - size_loss]
    
    elif deci_type == 'norm-L4':
        
        data = data[: ,: ,half_width:data.shape[-2] - half_width, half_width:data.shape[-1] - half_width]
        
        data = downsampling(data, scale = (4,4))
        
        return data

# The downscaling of aster needs to be reviewed since we didnt remove the pixels like with modis...
def downscale_Aster_to_coarse(
    data: np.ndarray,
    factor: float = 926.25/90,
    mtf: float = 0.1,
    padding="same",
    hkw: None = None,
):
    """
    Downsample patches with proper aliasing filtering
    
    Note: Code slightly modified from previous function. It should be applied to the Aster data but not transformed into its starting crs. We pass from array to tensor in order to be able to call torch's functions.
    """
    data_tensor = torch.Tensor(data).unsqueeze(0).unsqueeze(0)
    
    # Generate psf kernel for target MTF
    psf_kernel = torch.tensor(
        generate_psf_kernel(1.0, factor, mtf, hkw), device=data_tensor.device, dtype=data_tensor.dtype)
    
    half_width = int((psf_kernel.shape[-1] - 1) / 2)
        
    # Convolve data with psf kernel
    data = torch.nn.functional.pad(data_tensor, (half_width, half_width, half_width, half_width), mode="reflect")
    
    data = torch.nn.functional.conv2d(
        data,
        psf_kernel[None, None, :, :].expand(data.shape[1], -1, -1, -1),
        groups=data.shape[1],
        padding=padding,
    )
    # Downsample with nearest neighbors
    data = torch.nn.functional.interpolate(
        data, scale_factor=1 / factor, mode="bicubic"
    )
    
    return data.numpy()[0,0,:,:]


def downscale_Aster_to_fine(
    data: np.ndarray,
    factor: float = 231.656/90,
    mtf: float = 0.1,
    padding="same",
    hkw: None = None,
):
    """
    Downsample patches with proper aliasing filtering
    
    Note: Code slightly modified from previous function. It should be applied to the Aster data but not transformed into its starting crs. We pass from array to tensor in order to be able to call torch's functions.
    """
    data_tensor = torch.Tensor(data).unsqueeze(0).unsqueeze(0)
    
    # Generate psf kernel for target MTF
    psf_kernel = torch.tensor(
        generate_psf_kernel(1.0, factor, mtf, hkw), device=data_tensor.device, dtype=data_tensor.dtype)
    
    half_width = int((psf_kernel.shape[-1] - 1) / 2)
        
    # Convolve data with psf kernel
    data = torch.nn.functional.pad(data_tensor, (half_width, half_width, half_width, half_width), mode="reflect")
    
    data = torch.nn.functional.conv2d(
        data,
        psf_kernel[None, None, :, :].expand(data.shape[1], -1, -1, -1),
        groups=data.shape[1],
        padding=padding,
    )
    # Downsample with nearest neighbors
    data = torch.nn.functional.interpolate(
        data, scale_factor=1 / factor, mode="bicubic"
    )
    
    return data.numpy()[0,0,:,:]


def get_output_ftm(
    data: np.ndarray,
    factor: float = 4,
    mtf: float = 0.1,
    padding="same",
    hkw: None = None,
):
    """
    Downsample patches with proper aliasing filtering
    """
    # Generate psf kernel for target MTF
    psf_kernel = torch.tensor(
        generate_psf_kernel(1.0, factor, mtf, hkw), device=data.device, dtype=data.dtype
    )
    
    half_width = int((psf_kernel.shape[-1] - 1) / 2)
    
    # Convolve data with psf kernel
    data = torch.nn.functional.pad(data, (half_width, half_width, half_width, half_width), mode="reflect")
    
    data = torch.nn.functional.conv2d(
        data,
        psf_kernel[None, None, :, :].expand(data.shape[1], -1, -1, -1),
        groups=data.shape[1],
        padding=padding,
    )
    
    return data[:,:,half_width:data.shape[-2] - half_width,half_width:data.shape[-1] - half_width]



def get_output_ftm_ds_C(
    data: np.ndarray,
    factor: float = 4,
    mtf: float = 0.1,
    padding="same",
    hkw: None = None,
):
    """
    Downsample patches with proper aliasing filtering
    """
    
    data = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    
    # Generate psf kernel for target MTF
    psf_kernel = torch.tensor(
        generate_psf_kernel(1.0, factor, mtf, hkw), device=data.device, dtype=data.dtype
    )
    
    half_width = int((psf_kernel.shape[-1] - 1) / 2)
    
    # Convolve data with psf kernel
    data = torch.nn.functional.pad(data, (half_width, half_width, half_width, half_width), mode="reflect")
    
    data = torch.nn.functional.conv2d(
        data,
        psf_kernel[None, None, :, :].expand(data.shape[1], -1, -1, -1),
        groups=data.shape[1],
        padding=padding,
    )
    
    return data[0,0,half_width:data.shape[-2] - half_width,half_width:data.shape[-1] - half_width].numpy()

###############################################################################
# Defining the gssim
import scipy as sp
import numpy as np
import skimage as ski

# This here is basically the ssim from skimage but modified in order 
# to compute the GSSIM
def gssim(
    im1,
    im2,
    win_size=7,
    data_range=None,
    grad_comp_type = 1
):
    # Grad comp type = 1 --> grad based on sobel, 2 = low freq filtering.
    
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    
    filters = [ [[-1, 0, 1],[-2, 0 , 2],[-1, 0, 1]],
                [[-1, -2, -1],[0, 0, 0],[1, 2, 1]]
        ]
    
    # Gradients ('valid' since otherwise the values at the limits are diverging)
    f0 = sp.signal.convolve2d(im1, filters[0], mode = 'valid')
    f1 = sp.signal.convolve2d(im1, filters[1], mode = 'valid')
    g0 = sp.signal.convolve2d(im2, filters[0], mode = 'valid')
    g1 = sp.signal.convolve2d(im2, filters[1], mode = 'valid')
    
    # Magnitudes
    f_mag = np.sqrt(f0**2 + f1**2)
    g_mag = np.sqrt(g0**2 + g1**2)
    
    # To account for the valid keyword
    im1 = im1[1:im1.shape[0] - 1, 1:im1.shape[1] - 1]
    im2 = im2[1:im2.shape[0] - 1, 1:im2.shape[1] - 1]

    K1 = 0.01
    K2 = 0.03
    # sigma = 1.5

    use_sample_covariance = True

    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError(
            'win_size exceeds image extent. '
            'Either ensure that your images are '
            'at least 7x7; or pass win_size explicitly '
            'in the function call, with an odd value '
            'less than or equal to the smaller side of your '
            'images. If your images are multichannel '
            '(with color channels), set channel_axis to '
            'the axis number corresponding to the channels.'
        )

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    ndim = im1.ndim

    filter_func = sp.ndimage.uniform_filter
    filter_args = {'size': win_size}

    NP = win_size**ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute means (for the luminance term, uses only the starting images in gssim)
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # # compute variances and covariances (for contrast and structure, uses only the gradients in gssim)
    uxx = filter_func(f_mag * f_mag, **filter_args)
    uyy = filter_func(g_mag * g_mag, **filter_args)
    uxy = filter_func(f_mag * g_mag, **filter_args)
    vx = cov_norm * (uxx - filter_func(f_mag, **filter_args) ** 2)
    vy = cov_norm * (uyy - filter_func(g_mag, **filter_args) ** 2)
    vxy = cov_norm * (uxy - filter_func(f_mag, **filter_args) * filter_func(g_mag, **filter_args))

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux**2 + uy**2 + C1,
        vx + vy + C2,
    )
    
    # D = B1 * B2
    # S = (A1 * A2) / D

    L = A1 / B1 
    C = (2* np.sqrt(vx) * np.sqrt(vy) + C2) / B2
    S = (vxy + C2) / (np.sqrt(vx) * np.sqrt(vy) + C2/2)
    
    ssim = L*C*S
    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    mssim = ski.util.arraycrop.crop(ssim, pad).mean(dtype=np.float64)

    return mssim
