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
for file in sort_files:
    with xr.open_dataset(file) as open_data:
        print('Data sucessfully opened.')

        # access snow rain fraction data
        SR_sans_time = open_data['SR']

        # access time dimension for entire dataset
        raw_time = open_data['Times'].values.astype(str)

        # add Time as a recongizable dimension
        dt_index = pd.to_datetime(raw_time, format='%Y-%m-%d_%H:%M:%S')
        SR = SR_sans_time.assign_coords(Time = dt_index)
        print('Time coordinate assigned.')

        # create date mask for October 1 - April 1 
        date_mask = (SR['Time'].dt.month.isin([10, 11, 12, 1, 2, 3])) | ((SR['Time'].dt.month == 4) & (SR['Time'].dt.day == 1))

        # apply mask to dataset
        SR_slice = SR.sel(Time = date_mask)
        print('Date mask applied.')

        # move data to list
        results.append(SR_slice.load())
        print(f'{idx} of {len(sort_files)} added to list!')

# combine all of results into one dataset
final_ds = xr.concat(results, dim = 'Time')
print('Files successulyy concatonated.')

# save as netcdf
save_name = 'SR_slice.nc'
save_me = final_ds.to_netcdf(save_name)
print(f'Data saved to {save_name}.')
