#%%
"""
Author: Sydney Smith
Date Created: August 18, 2026
"""

import dask.array as dsa
import glob
from pathlib import Path
import pooch
import sys
import xarray as xr
import zarr



# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from old.temporal_chunks import loc_mask

var = 'tmmn'
# Define paths for observation and model data
obs_path = current_dir / 'gridMET' / var
model_path = current_dir / 'daily' / var


# Open datasets lazily using dask chunks
obs = xr.open_mfdataset(glob.glob(str(obs_path / '*.nc')), combine = 'nested', concat_dim = 'time').sel(time = slice('1985-01-01', '2001-12-31'))
print(obs)
print(obs.tmmn.data)
hist = xr.open_mfdataset(glob.glob(str(model_path / '*.nc')), combine = 'nested', concat_dim = 'time').sel(time = slice('1985-01-01', '2014-12-31'))
future_ds = xr.open_dataset(glob.glob(str(model_path / '*.nc')), combine = 'nested', concat_dim = 'time').sel(time = slice('2015-01-01', '2099-12-31'))


# # %%

# ds = xr.open_mfdataset(
#     glob.glob(str(current_dir / 'gridMET' / 'tmmn' / '*')), 
#     combine="nested", 
#     concat_dim="time", 
# )

# probe = ds['lat'].sel(south_north = 0, east_west = 0)

# # %%
# # create initial chunk structure
# ds = ds.chunk({"time": 100})
# ds.tmmn.encoding = {}  # helps when writing to zarr

# # %%
# ds.tmmn.data

# # %%

# ds.to_zarr('min_temp.zarr', mode = 'w')


# source_group = zarr.open('min_temp.zarr')
# print(source_group.tree())


# source_array = source_group['tmmn']
# print(source_array.info)

# # %%
# from rechunker import rechunk

# target_chunks = (365, 10, 10)
# max_mem = "500MB"

# target_store = "air_rechunked.zarr"
# temp_store = "air_rechunked-tmp.zarr"

# array_plan = rechunk(
#     source_array, target_chunks, max_mem, target_store, temp_store=temp_store
# )

# # %%
# result = array_plan.execute()
# result.chunks

# # %%
# from dask.diagnostics import ProgressBar

# with ProgressBar():
#     array_plan.execute()
# # %%
