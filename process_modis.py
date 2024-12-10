#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Script used for processing the .hdf files into pairs of patches saved as .tif obtained from 
splitting the big images.(1200x1200) and (4800x4800)
The files are then stored at ./data/geotiff/product_name .
This script also outputs:
    pairs.csv: file containing all the .tiff pairs.
    pairs_error.png: image presenting the error in the georeference between LST and NDVI.

Example
-------
>>> python process_modis.py --coverage 0.0

Else
----
@author: Romuald Ait Bachir
"""

import matplotlib.pyplot as plt

import os 
import glob
import time

import utils as us
import numpy as np
import pandas as pd
from skimage import morphology as morph

from osgeo import gdal

from argparse import ArgumentParser


def process_MOD11A1(hdf_name: str,
                    save_path: str,
                    window_size: int = 64,
                    coverage: float = 0.0,
                    time: str = 'day') -> None:
    """
    Description
    -----------
    Function extracting the small patches from the LST big image.These patches are 
    extracted if and only if there are no bad (water, cloud, bad quality) pixels inside 
    the patch.
    
    At the moment, these function only processes the day LST.
    
    ## 
    What to when we decide to also do the night LST?? 
    --> Just do both on their own, take the NDVI and make 2 pairs.csv file.
    --> Modify the datasetA to read both .csv and still have a 50/50 LST NDVI
    --> Modify the datasetB to concatenate both .csv in one dataframe.
    
    Parameters
    ----------
    hdf_name : string
        Path to the MOD11A1 hdf file.
        Typical path: ./data/hdf_files/MOD09GQ.061/
    save_path : string
        Path leading to the geotiff saving directory.
        Typical path: ./data/geotiff/MOD11A1.061/
    window_size : tuple, optional
        Size of the patches taken in the LST. The default value is 64.
    coverage : float, optional
        Percentage of the image covered by clouds or non quality pixels. 
        The default value is 0.0. 
    time : str, optional
        String defining the LST image that we want. Choose between 'day' and
        'night'. The default value is 'day'.
        
    """
    
    # Making the directory where the LST patches will be saved.
    os.makedirs(save_path, exist_ok=1)
    
    # Extracting the names of the files.
    name = hdf_name.split("/")[-1][:-3]
    
    cnt1 = 0 # This counter will permit the association of a "serial number" to an image due to the unique path taken inside the image
    
    # Reading the LST .hdf file
    LST_K, LST_QC, cols, rows, projection, geotransform = us.read_LST(hdf_name, time)
    
    for (x, y, extract_lst, extract_qc) in us.split(LST_K, LST_QC, (window_size, window_size)):
        cnt1 += 1
        if extract_lst.shape[0] == 64 and extract_lst.shape[1] == 64:
            
            # 1st condition: Not including the patches containing at least a ratio
            # of 0 K temperature. This condition, on 3 years leads to ~ 15000 images 
            # for day.
            temp_cond = np.zeros((extract_lst.shape[0], extract_lst.shape[1]))
            temp_cond[extract_lst == 0.0] = 1
            
            # 2nd condition attempt by watching the QC: Let's remove the patches 
            # having other quality data ie second lowest bit at 1
            QC_bits = np.zeros((extract_qc.shape[0], extract_qc.shape[1]))

            for i in range(64):
                for j in range(64):
                    QC_bits[i,j] = np.unpackbits(extract_qc[i,j])[-1]
            
            temp_cond = temp_cond + QC_bits
            temp_cond[temp_cond > 0.0] = 1
            # ONLY A 1000 when adding this threshold...
            # THIS CONDITION IS WAY TOO HARSH AND THE DATASET IS CONSEQUENTLY 
            # REDUCED. 
            # Condition on the coverage
            if np.sum(np.sum(temp_cond)) <= coverage * window_size**2:
    
                fname = os.path.join(os.getcwd(),save_path,name+str(cnt1)+'.tiff')
                
                # Updating the geotransform parameter for it to fit to the patches. 
                # X_init , Y_init --> X_patch, Y_patch. 
                # 1st: y term and 2nd : x term should be negligeable
                new_geotransform = np.asarray(geotransform)
                new_geotransform[0] = geotransform[0] + x * geotransform[1] + y * geotransform[2]
                new_geotransform[3] = geotransform[3] + x * geotransform[4] + y * geotransform[5] 
                
                # Saving the processed data as GeoTiff file.
                us.save_GeoTiff(extract_lst, fname, projection, new_geotransform)

def process_MOD21A1D(hdf_name: str,
                      save_path: str,
                      window_size: int = 64,
                      coverage: float = 0.0) -> None:
    """
    Description
    -----------
    Function extracting the small patches from the LST big image.These patches are 
    extracted if and only if there are no bad (water, cloud, bad quality) pixels inside 
    the patch.
    
    At the moment, these function only processes the day LST.
    
    ## 
    What to when we decide to also do the night LST?? 
    --> Just do both on their own, take the NDVI and make 2 pairs.csv file.
    --> Modify the datasetA to read both .csv and still have a 50/50 LST NDVI
    --> Modify the datasetB to concatenate both .csv in one dataframe.
    
    Parameters
    ----------
    hdf_name : string
        Path to the MOD11A1 hdf file.
        Typical path: ./data/hdf_files/MOD09GQ.061/
    save_path : string
        Path leading to the geotiff saving directory.
        Typical path: ./data/geotiff/MOD11A1.061/
    window_size : tuple, optional
        Size of the patches taken in the LST. The default value is 64.
    coverage : float, optional
        Percentage of the image covered by clouds or non quality pixels. 
        The default value is 0.0. 
        
    """
    
    # Making the directory where the LST patches will be saved.
    os.makedirs(save_path, exist_ok=1)
    
    # Extracting the names of the files.
    name = hdf_name.split("/")[-1][:-3]
    
    cnt1 = 0 # This counter will permit the association of a "serial number" to an image due to the unique path taken inside the image
    
    # Reading the LST .hdf file. We can use the same function as MOD11A1
    LST_K, LST_QC, cols, rows, projection, geotransform = us.read_LST(hdf_name, 'day')
    
    for (x, y, extract_lst, extract_qc) in us.split(LST_K, LST_QC, (window_size, window_size)):
        cnt1 += 1
        if extract_lst.shape[0] == 64 and extract_lst.shape[1] == 64:
            
            # 1st condition: Not including the patches containing at least a ratio
            # of 0 K temperature. This condition, on 3 years leads to ~ 15000 images 
            # for day.
            temp_cond = np.zeros((extract_lst.shape[0], extract_lst.shape[1]))
            temp_cond[extract_lst == 0.0] = 1

            # Condition on the coverage
            if np.sum(np.sum(temp_cond)) <= coverage * window_size**2:
    
                fname = os.path.join(os.getcwd(),save_path,name+str(cnt1)+'.tiff')
                
                # Updating the geotransform parameter for it to fit to the patches. 
                # X_init , Y_init --> X_patch, Y_patch. 
                # 1st: y term and 2nd : x term should be negligeable
                new_geotransform = np.asarray(geotransform)
                new_geotransform[0] = geotransform[0] + x * geotransform[1] + y * geotransform[2]
                new_geotransform[3] = geotransform[3] + x * geotransform[4] + y * geotransform[5] 
                
                # Saving the processed data as GeoTiff file.
                us.save_GeoTiff(extract_lst, fname, projection, new_geotransform)


def sort_files(files: list):
    """
    Function sorting the list based on the year, the month and the cropped patch's number.
    """
    def get_year(f: str) -> str:
        return f.split('.')[-6][1:5]

    def get_day(f: str) -> str:
        return f.split('.')[-6][5:]
    
    def get_tile(f: str) -> int:
        return int(f.split('.')[-2])

    return sorted(files, key=lambda f: (get_year(f), get_day(f), get_tile(f)))


def find_corresponding_NDVI(NIRRed_folder: str,
                            cropped_folder: str,
                            out_folder: str,
                            window_size: int = 256,
                            time: str = 'day') -> None:
    """
    Description
    -----------
    Function finding the corresponding NIRRed file to LST crops, searching the same
    area corresponding to the LST patches based on the date and the crop number, computing and
    saving the NDVI and pairing the NDVI and LST files into a .csv which will later
    be used as a "database".
    
    Parameters
    ----------
    NIRRed_folder : string
        Path to the folder containing the files from MOD09GQ.
        Typical path: ./data/hdf_files/MOD09GQ.061/
    cropped_folder : string
        Path to the folder containing the results of the cropping of LST files.
        Typical path: ./data/geotiff/MOD11A1.061/
    out_folder : string
        Path to the folder where the NDVI tiffs will be saved.
        Typical path: ./data/geotiff/MOD09GQ.061/
    window_size : int, optional
        Size of the patches on the NDVI image. The default value is set to 256.
    time : str, optional
        String defining the LST image that we want. Choose between 'day' and
        'night'. The default value is 'day'.
    """
    os.makedirs(out_folder, exist_ok=1)

    pair_list = [] # List of tuple that will be potentially transformed into a .csv
    
    cropped_files = glob.glob(os.path.join(cropped_folder,"*.tiff"))
    cropped_files = sort_files(cropped_files) #In order to go only once inside the file per day
    
    # We mostly do the same things as in process_MOD11A1. The main difference comes with 
    # the pairing of files: we need to filters patches that will certainly not have a 
    # corresponding LST, etc... The code is made to check this and is sort of optimized
    # to do things the less amount of time.
    dates_tiles = [(file.split('.')[-6][1:],file.split('.')[-2]) for file in cropped_files]
    
    d = {}
    for dt in dates_tiles:
        if not dt[0] in d.keys():
            d[dt[0]] = [dt[1]]
        else:
            d[dt[0]].append(dt[1])

    for key in d.keys():
        # Keys are the dates
        
        file_found = glob.glob(os.path.join(NIRRed_folder,"*A{}*.hdf".format(key)))
                
        if len(file_found) == 0:
            continue
        
        name = file_found[0].split("/")[-1][:-3]
        
        # Reading the NIR/Red file.
        Red, NIR, cols, rows, projection, geotransform = us.read_NIRRED(file_found[0])
        
        cnt1 = 0 # Counts the number of patch made
        cnt2 = 0 # Counts the number of values seen in dictionnary
        cnt3 = 0 # Counts the number of files that were deleted based on a NDVI with diverging dynamic range
      
        for (x, y, ext_NIR, ext_Red) in us.split_NIRRed(NIR, Red , (window_size, window_size)):
                
                cnt1 += 1
                                
                if str(cnt1) in d[key]:

                    fname = os.path.join(os.getcwd(),out_folder,name+str(cnt1)+'.tiff')
                    
                    # The NDVIs with zero division are not saved and the corresponding LST is removed
                    if (0.0 in ext_NIR + ext_Red):
                        
                        lst_name = glob.glob(os.path.join(cropped_folder,"*.*{}*.*.{}.tiff").format(key,cnt1))[0]

                        os.remove(lst_name)
                        
                        cnt3 += 1
                        continue 
                    
                    ext_ndvi = us.compute_NDVI(ext_NIR, ext_Red)

                    # Some values in the NDVI are outside the [-1,1] range due 
                    # to a too low denominator
                    if True:
                        ext_ndvi[ext_ndvi>1] = 1
                        ext_ndvi[ext_ndvi<-1] = -1
                        
                    # Updating the geotransform
                    new_geotransform = np.asarray(geotransform)
                    new_geotransform[0] = geotransform[0] + x * geotransform[1] + y * geotransform[2]
                    new_geotransform[3] = geotransform[3] + x * geotransform[4] + y * geotransform[5] 
                    
                    # Saving the data into a GeoTiff
                    us.save_GeoTiff(ext_ndvi, fname, projection, new_geotransform)
                    
                    # Finding the names and checking (just in case) that they match
                    ndvi_name = os.path.relpath(fname, os.getcwd())
                    lst_name = os.path.relpath(glob.glob(os.path.join(cropped_folder,"*.*{}*.*.{}.tiff").format(key,cnt1))[0], os.getcwd())
                    
                    if ndvi_name.split(".")[-2] != lst_name.split(".")[-2]:
                        print(ndvi_name.split(".")[-2])
                        print(lst_name.split(".")[-2])
                        os.remove(ndvi_name)
                        os.remove(lst_name)
                        cnt3 += 1
                        
                    pair_list.append((lst_name, ndvi_name))
                    
                    cnt2 += 1
                    
                if cnt2 == len(d[key]):
                    break
    
    df = pd.DataFrame(pair_list, columns = ('LST', 'NDVI'))
    df.to_csv(os.path.join(os.getcwd(),"data/pairs_"+time+".csv"))
    print("Total Number of LST patches deleted due to faulty NDVI: {}".format(cnt3))


def add_water_masks(water_folder: str = "./data/hdf_files/MOD44W.061",
                    out_folder: str = "./data/geotiff/MOD44W.061",
                    csv_path: str = "./data/pairs_day.csv",
                    window_size: int = 256) -> None:
   
    os.makedirs(out_folder, exist_ok = True)
    
    df = pd.read_csv(csv_path)
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    
    data = df.to_numpy()
    
    new_data = np.zeros((data.shape[0], data.shape[1] + 1), dtype = object)
    new_data[:,0:2] = data
    
    element = np.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]],dtype = np.uint8)
    
    for idx, d in enumerate(data):
        print(idx)
        lst_name = d[0]
        end_name = lst_name.split('/')[-1]
    
        block_number = int(int(float(end_name.split('.')[-2])))
        year = end_name.split('.')[1][1:5]
        
        # Absolute path of the patch that will be saved.
        name = os.path.join(out_folder, "MOD44W."+year+'.'+str(block_number)+'.tiff')
        
        # Now we generate the patch.
        path_water_hdf = os.path.join('data','hdf_files','MOD44W.061')
        water_file = [os.path.join(path_water_hdf, f) for f in os.listdir(path_water_hdf) if os.path.isfile(os.path.join(path_water_hdf, f)) if not 'xml' in f and year in f][0]
        
        ndvi, _, _, projection, geotransform = us.read_GeoTiff(d[1])
        water, _, _, _, _ = us.read_MOD44W(water_file)

        cnt = 0
        for i in range(0, water.shape[0], window_size):
            for j in range(0, water.shape[1], window_size):
                cnt += 1 
                if cnt == block_number:
                    wat = water[j:j+window_size, i:i+window_size]
        
        wat = morph.dilation(wat, element)
        new_data[idx, 2] = name
        us.save_GeoTiff(wat, name, projection, geotransform)
        
    df_water = pd.DataFrame({'LST': new_data[:, 0], 'NDVI': new_data[:, 1], 'Water Mask': new_data[:, 2]})
    df_water.to_csv(os.path.join(os.getcwd(),"data/pairs_day.csv"))


def measure_Georeference(file: str):
    """
    Description
    -----------
    Function measuring the difference in georeference between the LST and the 
    NDVI files in the generated pair.csv file.
    
    The difference is measured using: delta = abs(deltaX) + abs(deltaY) 

    Parameters
    ----------
    file : str
        .csv file containing the pairs (LST, NDVI). One of the output of find_corresponding_NDVI.

    Returns
    -------
    df : pandas.DataFrame
        Transformed .csv table as a pandas.DataFrame with on each row the Georeferenced error.

    """
    # First two functions are defined in order to read the GeoTransform data and
    # compute the error between the LST data and NDVI data.
    def get_Geotransform(filename: str):
        return gdal.Open(filename, gdal.GA_ReadOnly).GetGeoTransform()
    
    def compute_geodiff(geopos1: list, geopos2: list):
        return abs((geopos1[0] - geopos2[0])) + abs((geopos1[3] - geopos2[3]))
    
    # Reading pairs.csv
    df = pd.read_csv(file)
    
    # Removing the first index column
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    
    # Reading the files and computing the error
    df = df.map(lambda x : get_Geotransform(x))
    df = df.apply(lambda x : compute_geodiff(x['LST'], x['NDVI']), axis = 1)
    return df


if __name__ == '__main__':
    # This script is quite 'fast' in comparison to the download process.
    # It should take between 10 to 20 minutes to process 3 years of data. 
    # Between 800 to 900 seconds. Note: This time estimate doesn't include the 
    # water mask.
    parser = ArgumentParser()
    parser.add_argument('--coverage', type = float,  default=0.00)
    parser.add_argument('--product_LST', type = str, default = 'MOD21A1D')
    parser.add_argument('--water_mask', type = bool, default = False)
    args = parser.parse_args()
   
    # Image extraction parameters
    coverage = args.coverage
    product_LST = args.product_LST
    water_mask = args.water_mask
    
    tm = time.time()
    
    # Making a results folder
    results_path = "./results/dataset"
    os.makedirs(results_path, exist_ok=True)

    ######### For day images ##################################################
    print("Processing of the day images...")
    
    if product_LST == 'MOD11A1': 
        MOD11A1_path = os.path.join(os.getcwd(), "data/hdf_files/MOD11A1.061")
        MOD11A1_files = glob.glob(os.path.join(MOD11A1_path,"*.hdf"))
    
        # Path to the folder where the LST tiffs will be found
        # should modify and add /day
        save_path = os.path.join(os.getcwd(),"data/geotiff/MOD11A1.061/day")
        
        print("Starting the extraction of the patches inside the LST files.")
        
        for file in MOD11A1_files:
            process_MOD11A1(file, save_path, 64, coverage)
    
    elif product_LST == 'MOD21A1D': 
        MOD21A1D_path = os.path.join(os.getcwd(), "data/hdf_files/MOD21A1D.061")
        MOD21A1D_files = glob.glob(os.path.join(MOD21A1D_path,"*.hdf"))
    
        # Path to the folder where the LST tiffs will be found
        # should modify and add /day
        save_path = os.path.join(os.getcwd(),"data/geotiff/MOD21A1D.061/day")
        
        print("Starting the extraction of the patches inside the LST files.")
        
        for file in MOD21A1D_files:
            process_MOD21A1D(file, save_path, 64, coverage)
    
    print("{} LST files were extracted".format(len(glob.glob(os.path.join(save_path,'*.tiff')))))

    # Path to the folder containing the downloaded NIR/Red hdf files
    NIRRed_folder = os.path.join(os.getcwd(),"data/hdf_files/MOD09GQ.061")
    
    # Path to the folder where the NDVIs will be found
    out_folder = os.path.join(os.getcwd(),"data/geotiff/MOD09GQ.061/day")
    
    # Finding the corresponding NDVI, saving the patches and storing the dataset
    # as a .csv file called pairs.csv
    find_corresponding_NDVI(NIRRed_folder, save_path, out_folder, 256)
    
    # Adding the water masks
    if water_mask:
        add_water_masks()
    
    # Reading pairs.csv and measuring all the errors in the georeference in the pairs.
    df = measure_Georeference("./data/pairs_day.csv")
    
    ax = df.plot.hist(x= 'Error', y= 'Count', bins = 30, figsize = (14,7), title = 'Histogram of the difference in georeference in a pair LST/NDVI')
    fig = ax.get_figure()
    fig.savefig(os.path.join(results_path,'pairs_day_error.png'))
    # This figure outputs the repartition of the georeference error happening 
    # when making our pairs based on the patch number. 
    # For the day images, the error is really close to zero so there is no problem.

    
    # ######### For night images ################################################
    # print("Processing of the night images...")
    
    # MOD11A1_path = os.path.join(os.getcwd(), "data/hdf_files/MOD11A1.061")
    # MOD11A1_files = glob.glob(os.path.join(MOD11A1_path,"*.hdf"))

    # # # Path to the folder where the LST tiffs will be found
    # save_path = os.path.join(os.getcwd(),"data/geotiff/MOD11A1.061/night")
    
    # print("Starting the extraction of the patches inside the LST files.")
    
    # for file in MOD11A1_files:
    #     process_MOD11A1(file, save_path, 64, coverage, time = 'night')
    
    # print("{} LST files were extracted".format(len(glob.glob(os.path.join(save_path,'*.tiff')))))

    # #Path to the folder containing the downloaded NIR/Red hdf files
    # NIRRed_folder = os.path.join(os.getcwd(),"data/hdf_files/MOD09GQ.061")
    
    # # Path to the folder where the NDVIs will be found
    # out_folder = os.path.join(os.getcwd(),"data/geotiff/MOD09GQ.061/night")
    
    # # Finding the corresponding NDVI, saving the patches and storing the dataset
    # # as a .csv file called pairs.csv
    # find_corresponding_NDVI(NIRRed_folder, save_path, out_folder, 256, 'night')
    
    # ## Note: This is absolutely useless for night images since they are localized 
    # ## at the same place as the day without accounting for the shift...
    # # # FIND CORRESPONDING NDVI SHOULD BE USED BUT THE ERROR SHOULD ALSO BE MEASURED!!
    # # df = measure_Georeference("./data/pairs_night.csv")
 
    # # ax = df.plot.hist(x= 'Error', y= 'Count', bins = 30, figsize = (14,7), title = 'Histogram of the difference in georeference in a pair LST/NDVI')
    # # fig = ax.get_figure()
    # # fig.savefig(os.path.join(results_path,'pairs_night_error.png'))
    
    print("Time taken in total: {} seconds".format(time.time() - tm))
    
###############################################################################
