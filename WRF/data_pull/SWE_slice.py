# %%
"""
Author: Sydney Smith
Date: June 18, 2026
"""

import glob
import os
import pandas as pd
import xarray as xr


path = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/'
grab_files = os.path.join(path, '*', 'wrfout_d03*')
sort_files = sorted(glob.glob(grab_files))
print('File names pulled and sorted.')
file = xr.open_dataset(sort_files[0])

# %%
# create list to store results in
results = []

# loop through each file individual to prevent memory from being overloaded
for idx, file in enumerate(sort_files):
    try:
        with xr.open_dataset(file) as open_data:
            print(f'{file} sucessfully opened.')

            # access snow data (SWE)
            SWE_sans_time = open_data['SNOW']

            # access time dimension for entire dataset
            raw_time = open_data['Times'].values.astype(str)

            # add Time as a recongizable dimension
            dt_index = pd.to_datetime(raw_time, format='%Y-%m-%d_%H:%M:%S')
            SWE = SWE_sans_time.assign_coords(Time = dt_index)
            print('Time coordinate assigned.')

            # create date mask for October 1 - April 1 
            date_mask = (SWE['Time'].dt.month.isin([10, 11, 12, 1, 2, 3])) | ((SWE['Time'].dt.month == 4) & (SWE['Time'].dt.day == 1))

            # apply mask to dataset
            SWE_slice = SWE.sel(Time = date_mask)
            print('Date mask applied.')

            # move data to list
            results.append(SWE_slice.load())
            print(f'{idx} of {len(sort_files)} added to list!')

    except Exception as e:
        print(f'FAILED FILE: {file}')
        print(f'ERROR: {e}')
        continue # Skip the bad file and keep processing others

# combine all of results into one dataset
final_ds = xr.concat(results, dim = 'Time')
print('Files successulyy concatonated.')

# save as netcdf
save_name = 'SWE_slice.nc'
save_me = final_ds.to_netcdf(save_name)
print(f'Data saved to {save_name}.')

# %%
