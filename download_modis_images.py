#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Script used for downloading the modis files for the products MOD11A1 and MOD09GQ 
over a time period. The products of interest could be easily modified by changing 
the short_names variable inside the main.

Example
-------
python download_modis_images.py --username projet3a --password Projet3AIMT --start_date 2023-08-01 --stop_date 2023-08-08 --n_threads = 8

Else
----
@author: Romuald Ait Bachir
"""

from pymodis import downmodis

import os 
import utils as us

from argparse import ArgumentParser
import pandas
from datetime import timedelta, datetime
import pymp


def get_hdf(username: str,
            password: str, 
            tiles: str, 
            start: str, 
            stop: str, 
            short_name: str, 
            n_threads: int):
    """
    Description
    -----------
    Downloading the data (hdf, xml) at tile for the instrument short_name at 
    time period [start, stop].
    
    Let N be the number of days inside your time period. This function should 
    return 2*N + 1 files (N .hidf, N .xml and 1 txt).
    
    The .hdf file contains the data associated to the measurement of the sensor.
    The .xml file contains the metadata of the granule.
    
    Parameters
    ----------
    username: string
        Username to connect to the modis session.
    password: string
        Password to connect to the modis session.
    tiles: string
        Identifier of a tile. Format: "hHHvVV" with: HH: 00->35 and VV: 00->17.
        Look at the MODIS sinusoidal tile grid for more information about a region.
    start: string
        Starting date. Format: "YYYY-MM-DD".
    stop: string
        Ending date. Format: "YYYY-MM-DD".
    short_name: string
        Shortened name referencing a product used in modis. For example: MOD11A1.061 
        See the MODIS Naming Conventions at https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/modis-overview/ for more information.
    n_threads: int
        Number of threads on which to parallelize the download.
    
    Example
    -------
    >>> get_hdf("johnsmith","123456789","h12v04", "2020-01-01", "2020-02-01", "MOD11A1.061", 8)
    
    """
    
    # Making the tree structure. 
    path = os.path.join('data','hdf_files',short_name)
    os.makedirs(path, exist_ok=True)
    
    # Getting the dates for the time period and splitting it into n_threads chunks for adequate multi-threading.
    l_days = [t.strftime('%Y-%m-%d') for t in list(pandas.date_range(datetime.strptime(start, '%Y-%m-%d'),datetime.strptime(stop, '%Y-%m-%d')-timedelta(days=0),freq='d'))]
    
    # Need to split the work as much as possible. l_days is the limit at which 
    # it wouldn't be possible to fully parallelize the work
    if n_threads >= len(l_days):
        n_threads = len(l_days) 
    
    # Processing the dates to split them into n_threads chunk for parallelization.
    l_days1 = us.date_into_n_chunk(l_days, n_threads)
    
    # Parallelizing using pymp
    with pymp.Parallel(n_threads) as p:
           for split in p.range(0,n_threads):
               start = l_days1[split][0]
               stop = l_days1[split][1]
               
               try: 
                   print("Downloading for day {} to day {}".format(start,stop))
                   modisDown = downmodis.downModis(user=username,password=password,product=name, destinationFolder = path, tiles = tiles, today=start, enddate=stop)
                   modisDown.connect() # Connection to the server
                   modisDown.downloadsAllDay() # Download the data for the time period
               
               except:
                   print("Download Error, the data from tiles {} at date {} -> {} can't be downloaded".format(tiles ,start, stop))
        
        
if __name__ == "__main__":
    parser = ArgumentParser()
    # username and password should be removed if ever put somewhere online
    parser.add_argument('--username', type=str, default="projet3a")
    parser.add_argument('--password', type=str, default="Projet3AIMT")
    parser.add_argument('--start_date', type=str, default="2023-01-01")
    parser.add_argument('--stop_date', type=str, default="2023-01-02")
    parser.add_argument('--n_threads', type=int, default="12")
    # Default set to be small in order to have a simple and fast example.
    
    args = parser.parse_args()
    
    # Parameters
    user = args.username
    passw = args.password
    debut = args.start_date
    fin = args.stop_date
    n_threads = args.n_threads
    
    # Products names 
    short_names = ["MOD11A1.061", "MOD09GQ.061", "MOD21A1D.061", "MOD09GA.061", 'MOD44W.061']
    # Note: MOD44W.061 = 1 file/year so super fast to download.
    # Much slower for MOD11A1.061 and MOD21A1D.061
    # And even more slow for MOD09GQ.061 and MOD09GA.061
    # If you want the minimal working data, only download MOD21A1D.061 and 
    # MOD09GQ.061
    # !!! I also download MOD09GA RGB but don't process those later.
    
    # Tile of interest
    tiles = "h18v04" # centered on France, Italy, Spain, Germany
    
    # The download of the data is parallelized to accelerate the process
    for name in short_names:
        print("Downloading the Data of {}".format(name))
        get_hdf(username = user, password = passw, tiles = tiles, start = debut, stop = fin, short_name = name, n_threads = n_threads) 