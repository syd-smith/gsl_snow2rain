# %%
"""
Author: Sydney Smith
Date: June 18, 2026
"""

import glob
import os
import xarray as xr


path = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/'
grab_files = os.path.join(path, '*', 'wrfout_d03*')
sort_files = sorted(glob.glob(grab_files))

open_data = xr.open_mfdataset(sort_files, combine = 'by_coords', engine = 'netcdf4', chunks={'Time': 100})

SWE = open_data['SWE']

# create date mask for October 1 - April 1 
date_mask = (SWE['Time.month'].isin([10, 11, 12, 1, 2, 3])) | ((SWE['Time.month'] == 4) & (SWE['Time.day'] == 1))

# apply mask to dataset
SWE_month_slice = SWE.sel(Time = date_mask)

# save as netcdf
SWE_month_slice.to_netcdf('SWE_month_slice.nc')
