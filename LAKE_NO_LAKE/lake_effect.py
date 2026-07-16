# %%
"""
Author: Sydney Smith
Date: July 10, 2026
"""

import glob
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from pathlib import Path
import sys
from wrf import (getvar, ALL_TIMES)
import xarray as xr


# ==================================
# - Establish Relative File Path - 
# ==================================

try:
    current_file_directory = Path(__file__).resolve().parent
except NameError:
    current_file_directory = Path().resolve().parent
parent_directory = current_file_directory.parent
sys.path.append(str(parent_directory))

year = '1997'

jan1 = f'/uufs/chpc.utah.edu/common/home/strong-group8/lake/{year}_NL_new/wrfout_d03_{year}-01-01_00:00:00'
jan1_open = getvar(Dataset(jan1), 'RAINNC')

cut = 



#%%

feb_end = f'/uufs/chpc.utah.edu/common/home/strong-group8/lake/{year}_NL_new/wrfout_d03_{year}-03-03_00:00:00'
feb_end_open = getvar(Dataset(feb_end), 'RAINNC')

total_accum = feb_end_open - jan1_open
print(total_accum)

jan1_lake = f'/uufs/chpc.utah.edu/common/home/strong-group8/lake/{year}_LB/wrfoutput/wrfout_d03_{year}-01-01_00:00:00'
jan1_lake_open = getvar(Dataset(jan1_lake), 'RAINNC')

feb_end_lake = f'/uufs/chpc.utah.edu/common/home/strong-group8/lake/{year}_LB/wrfoutput/wrfout_d03_{year}-03-03_00:00:00'
feb_end_lake_open = getvar(Dataset(feb_end_lake), 'RAINNC')

total_lake = feb_end_lake_open - jan1_lake_open
print(total_lake)

# gridpoints for data slice
lat_min, lat_max = 40.476269, 40.813703
lon_min, lon_max = -111.800367, -111.584936

# slice data to cover only the Wasatch Front
NL_wasatch = total_accum.sel(XLONG = slice(lon_min, lon_max), XLAT = slice(lat_min, lat_max), method = 'nearest')
LB_wasatch = total_lake.sel(XLONG = slice(lon_min, lon_max), XLAT = slice(lat_min, lat_max), method = 'nearest')

# take the mean over the study region
NL_spatial_mean = NL_wasatch.mean(dim = {'south_north', 'west_east'})
LB_spatial_mean = LB_wasatch.mean(dim = {'south_north', 'west_east'})
anom_spatial_mean = anom.mean(dim = {'south_north', 'west_east'})

print(f'Without the lake, total precipitation from Jan 1 to Mar 3, {year} is {NL_spatial_mean} mm')
print(f'With the lake, total precipitation from Jan 1 to Mar 3, {year} is {LB_spatial_mean} mm')
print(f'The effect of lake effect precipitation for {year} is therefore {anom_spatial_mean} mm')





#%%







# NL = []
# LB = []

# for year in range(1997, 2001):
#     f_path = f'/uufs/chpc.utah.edu/common/home/strong-group8/lake/{year}_NL_new/'
#     NL_files = sorted(glob.glob(f_path + f'*d03_{year}-*'))
#     file_storage = []
#     for file in NL_files:
#         f_open = Dataset(file)
#         print(file)
#         print('XLAT' in f_open.variables)
#         file_storage.append(f_open)
#     snow_NL = getvar(file_storage, 'RAINNC', ALL_TIMES)
#     NL.append(snow_NL)

# for year in range(1997, 2001):
#     if year == 1997:
#         f_path = f'/uufs/chpc.utah.edu/common/home/strong-group8/lake/{year}_LB//wrfoutput/'
#     else:
#         f_path = f'/uufs/chpc.utah.edu/common/home/strong-group8/lake/{year}_LB/'
#     LB_files = sorted(glob.glob(f_path + f'*d03_{year}-*'))
#     file_storage = []
#     for file in LB_files:
#         f_open = Dataset(file)
#         print(file)
#         print('XLAT' in f_open.variables)
#         file_storage.append(f_open)
#     snow_LB = getvar(file_storage, 'RAINNC', ALL_TIMES)
#     LB.append(snow_LB)


# # %%
# # concatonate all items in the list into say object on the year dimension
# NL_combo = NL.concat(dim = 'Year')
# LB_combo = LB.concat(dim = 'Year')

# NL_diff = NL_combo.diff(dim = 'Time')
# LB_diff = LB_combo.diff(dim = 'Time')

# # take the mean of the year dimension 
# NL_mean = NL_diff.mean(dim = 'Year')
# LB_mean = LB_diff.mean(dim = 'Year')

# # gridpoints for data slice
# lat_min, lat_max = 40.476269, 40.813703
# lon_min, lon_max = -111.800367, -111.584936

# # slice data to cover only the Wasatch Front
# NL_wasatch = NL_mean.sel(XLONG = slice(lon_min, lon_max), XLAT = slice(lat_min, lat_max), method = 'nearest')
# LB_wasatch = LB_mean.sel(XLONG = slice(lon_min, lon_max), XLAT = slice(lat_min, lat_max), method = 'nearest')

# # calculate anomaly based on region specific data (if no lake is more then positive)
# anom = LB_mean - NL_mean

# # take the mean over the study region
# NL_spatial_mean = NL_wasatch.mean(dim = {'south_north', 'west_east'})
# LB_spatial_mean = LB_wasatch.mean(dim = {'south_north', 'west_east'})
# anom_spatial_mean = anom.mean(dim = {'south_north', 'west_east'})

# # %%

# fig, ax = plt.subplots(3, 1, figsize = (10, 8))
# ax[0].plot(NL_spatial_mean['Time'], NL_spatial_mean.values, label = 'No Lake', linewidth = 1, color = 'red')
# ax[1].plot(LB_spatial_mean['Time'], LB_spatial_mean.values, label = 'Lake', linewidth = 1, color = 'green')
# ax[2].plot(anom_spatial_mean['Time'], anom_spatial_mean.values, label = 'Anomaly', linewidth = 1, color = 'black')
    

# # %%
