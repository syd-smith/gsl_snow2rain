# %%
"""
Author: Sydney Smith
Date: June 23, 2026
"""

import glob
import os
import pandas as pd
import xarray as xr


path = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/'
grab_files = os.path.join(path, '*', 'wrfout_d03*')
sort_files = sorted(glob.glob(grab_files))
print('File names pulled and sorted.')

# create list to store results in
results = []

# loop through each file individual to prevent memory from being overloaded
for idx, file in enumerate(sort_files):
    with xr.open_dataset(file) as open_data:
        print('Data sucessfully opened.')

        # access terrain elevation data
        HGT_sans_time = open_data['HGT']

        # access time dimension for entire dataset
        raw_time = open_data['Times'].values.astype(str)

        # add Time as a recongizable dimension
        dt_index = pd.to_datetime(raw_time, format='%Y-%m-%d_%H:%M:%S')
        HGT = HGT_sans_time.assign_coords(Time = dt_index)
        print('Time coordinate assigned.')

        # create date mask for October 1 - April 1 
        date_mask = (HGT['Time'].dt.month.isin([10, 11, 12, 1, 2, 3])) | ((HGT['Time'].dt.month == 4) & (HGT['Time'].dt.day == 1))

        # apply mask to dataset
        HGT_slice = HGT.sel(Time = date_mask)
        print('Date mask applied.')

        # move data to list
        results.append(HGT_slice.load())
        print(f'{idx} of {len(sort_files)} added to list!')

# combine all of results into one dataset
final_ds = xr.concat(results, dim = 'Time')

# save as netcdf
save_name = 'HGT_slice.nc'
save_me = final_ds.to_netcdf(save_name)
print(f'Data saved to {save_name}.')
