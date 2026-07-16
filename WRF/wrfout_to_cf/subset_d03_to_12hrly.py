"""
Subset the really large d03 files to every...




"""
#%% Imports
import numpy as np
import xarray as xr
import os


#%% Variable Preset

START_YR = 2003
END_YR = 2003
YEARS = np.arange(START_YR,END_YR+1)

#file location
WRF_DIR = "/uufs/chpc.utah.edu/common/home/strong-group6/HIMAT/WRF_output/"

#directory path to where the figure will be saved
# SAVE_DIR = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/himat/jgr_wrf/"
SAVE_DIR = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/himat/redo_d03_2003_07_21/"

NEST = "d03"

SUBSET_INTERVALS = 16


#%% Load data


for yr in YEARS:
    file_path = WRF_DIR + str(yr) + "/output/"

    files = os.listdir(file_path)
    ncfiles = sorted([i for i in files if f"wrfout_{NEST}_" in i])
    ncfiles = [i for i in ncfiles if "." not in i] # remove any bad files
    
    for filex in range(len(ncfiles)):
        print(ncfiles[filex])
        
        #open netCDF file (ncread)
        ncfile = xr.open_dataset(file_path + ncfiles[filex])
        
        # get the time values
        time = ncfile['Times']
        
        # get the number of hours
        num_hours = len(time)
        print(num_hours)
        
        # define the time indicies for splitting the data
        # intervals = np.arange(0, num_hours, 12)
        intervals = np.round(np.linspace(0, num_hours, num=SUBSET_INTERVALS)).astype('int')
        
        count = 0
        
        # loooooop
        for i in range(len(intervals)-1):
            
            start_index = intervals[i]
            end_index   = intervals[i+1]
            
            # subset the dataset
            subset_ncfile = ncfile.isel(Time=slice(start_index, end_index))
            
            count += len(subset_ncfile['Times'])
        
            # create name of new output file
            date_name = str(subset_ncfile['XTIME'].values[0])
            date_name = date_name[0:10] + "_" + date_name[11:19]
            
            # save file
            path = f"{SAVE_DIR}{yr}/wrfout_{NEST}_{date_name}"
            subset_ncfile.to_netcdf(path)
            
        print(count)
        
        
        
        
#%% test if all the dates are there


START_YR = 2000
END_YR = 2015
YEARS = np.arange(START_YR,END_YR+1)

#file location
WRF_DIR = "/uufs/chpc.utah.edu/common/home/strong-group7/savanna/himat/jgr_wrf_cf_compress/d03/"

times = np.array([], dtype='datetime64')

for yr in YEARS:
    
    files = os.listdir(WRF_DIR)
    yrx = str(yr)
    ncfiles = sorted([i for i in files if f"wrfout_{NEST}_" + yrx in i])
    
    for filex in range(len(ncfiles)):
    
        #open netCDF file (ncread)
        ncfile = xr.open_dataset(WRF_DIR + ncfiles[filex])
    
        # get the time values
        time = np.array(ncfile['time'])
        
        times = np.concatenate((times, time))
        
        
tt = np.unique(times)

print(f"{NEST} has " + len(tt) + " hours")
        
        
    
        