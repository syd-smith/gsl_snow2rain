"""
Author: Sydney Smith
Date Created: August 18, 2026
"""

import dask.array as dsa
from dask.diagnostics import ProgressBar
import glob
import metpy.calc as mpcalc
from pathlib import Path
import pooch
from rechunker import rechunk
import sys
import xarray as xr
import zarr

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

"""
Practice with rechunking using Dask and Zarr based on the toy example from Rechunker.
https://rechunker.readthedocs.io/en/latest/tutorial.html#Toy-Example
"""

# Open practice data
ds = xr.open_mfdataset(
    glob.glob(str(current_dir / 'gridMET' / 'tmmn' / '*')), 
    combine = 'nested', 
    concat_dim = 'time;, 
)

# Select data slice
probe = ds['lat'].sel(south_north = 0, east_west = 0)

# Create initial chunk structure
ds = ds.chunk({"time": 100})
ds.tmmn.encoding = {}  # helps when writing to zarr

# Print visual of data
ds.tmmn.data

# Convert to Zarr
ds.to_zarr('min_temp.zarr', mode = 'w')

# Open Zarr data
source_group = zarr.open('min_temp.zarr')
print(source_group.tree()) # Prints tree of dimensions

# Select temperature data
source_array = source_group['tmmn']
print(source_array.info) # Prints info about array

# Define target inputs for chunking
target_chunks = (365, 10, 10)
max_mem = "500MB"
target_store = "air_rechunked.zarr"
temp_store = "air_rechunked-tmp.zarr"

# Create rechunk object
array_plan = rechunk(
    source_array, target_chunks, max_mem, target_store, temp_store=temp_store
)

# Execute rechunking via object defined above
result = array_plan.execute()
result.chunks

# Display with progress bar
with ProgressBar():
    array_plan.execute()

