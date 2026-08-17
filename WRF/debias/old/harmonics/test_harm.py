# %%
"""
Author: Sydney Smith
Date Created: July 28, 2026
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
import sys
import xarray as xr

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from temporal_chunks import harmonic_n, loc_mask

# Define variable of interest
var = 'tmmn'

# Load entire year worth of climatological daily bias data (raw noisy datapoints)
climate_dir = parent_dir / 'climate_avg' / var
climate_files = sorted(glob.glob(f'{str(climate_dir)}/climate_avg_{var}*.nc'))
ds = xr.open_mfdataset(climate_files, combine = 'nested', concat_dim = 'time').load() # Load noise into RAM and strip from other files to save space

# Set exmaple lat and lon values
lat_min, lat_max = 40.12, 40.13
lon_min, lon_max = -110.09, -110.04

# Mask data to specific location
noise_masked = loc_mask(ds, 'other', lat_min, lat_max, lon_min, lon_max).isel(lat = 0, lon = 0).squeeze()


# %%
def harmonic_formula(order):
    """
    Harmonic function custom to given order (based on number of coefficents passed in *coeffs). 
    This function outputs a formula to pass to statsmodels formula API. 
    """
    # *coeffs: captures any number of coefficients (a1, b1, a2, b2, a3, b3...)
    # omega: fundamental frequency of the entire dataset window (a year worth of daily data)
    omega = '(2 * np.pi * d)/365'
    # TODO: 
    # TODO: error if t isnt measured in days because then theres a mismatch with omega

    res = 'y_var ~ '
    for i in range(1, order + 1):
        res += f'np.cos({i} * {omega}) + np.sin({i} * {omega})'
        
        # Add a plus sign between terms for each order
        if i != order:
            res += ' + '

    return res
    # TODO: check what time values are set to in the dataset (should be numerical offset relative to start date of the dataset)

def fit_harmonic(y, d, order):
    # Pack the 1D arrays into a dictionary for patsy/formula API
    data_dict = {'y_var': y, 'd': d}
    
    # Define your formula (you can swap this with your dynamic harmonic_n function)
    formula = harmonic_formula(order = order)
    
    # Fit model using the formula API
    model = smf.ols(formula = formula, data = data_dict).fit()
    
    # model.params returns a pandas Series (Intercept, cos, sin)
    return model.params.values

test = fit_harmonic(noise_masked[var], noise_masked['time'].dt.dayofyear, 1)

# %%
order = 3
expected_num_coeffs = (2 * order) + 1
ds['day_of_year'] = ds['time'].dt.dayofyear

# Use xarray's apply_ufunc identically to before
coeffs = xr.apply_ufunc(
    fit_harmonic, # function being used
    ds[var], ds['day_of_year'], # args called in the function
    kwargs = {'order': order}, # unpack any other args in function
    input_core_dims = [['time'], ['time']],
    output_core_dims = [['lat'], ['lon'], ['coeff']],
    vectorize = True,
    dask = 'parallelized',
    output_dtypes = [float],
    dask_gufunc_kwargs = {'output_sizes': {'coeff': expected_num_coeffs}}
)

'y_var ~ np.cos(1 * (2 * np.pi * d)/365) + np.sin(1 * (2 * np.pi * d)/365) + np.cos(2 * (2 * np.pi * d)/365) 
+ np.sin(2 * (2 * np.pi * d)/365) + np.cos(3 * (2 * np.pi * d)/365) + np.sin(3 * (2 * np.pi * d)/365)'

#%%

# Rename coeffs dimensions to match your model's grid layout if needed
coeffs = coeffs.rename({'lat': 'south_north', 'lon': 'east_west'})

# Now assign the 2D coordinates safely
coeffs = coeffs.assign_coords({
    'lat': ds['lat'],
    'lon': ds['lon']
})

# 1. Select a specific grid cell (e.g., first lat/lon or a specific location)
# If your dataset has spatial dims, pick one: cell_coeffs = coeffs.isel(lat=0, lon=0)
# Or if it's a single time-series:
cell_coeffs = loc_mask(coeffs, 'other', lat_min, lat_max, lon_min, lon_max)
cell_data = noise_masked.isel(south_north=0, east_west=0).squeeze()
cell_data = cell_data[var].squeeze()  # Ensure it's 1D for plotting

# 2. Extract the intercept (index 0) as a clean float scalar using xarray selection
intercept = cell_coeffs.isel(coeff=0).values.item()

# 3. Reconstruct the fitted values over a 365-day range
d_plot = np.linspace(1, 365, 365)
y_pred = np.full_like(d_plot, intercept)

omega = (2 * np.pi) / 365

# 4. Plot the original data vs the multi-order harmonic fit
plt.figure(figsize = (10, 5))

# Loop through each order n (starting at 1)
for n, color in zip(range(1, order + 1), plt.cm.viridis(np.linspace(0, 1, order))):
    # Dynamically select the exact cosine and sine coefficient indices for order n
    # (Cosine is typically at odd positions, Sine at even positions)
    a_n = float(cell_coeffs.isel(coeff = (2 * n - 1)).values[0][0])
    b_n = float(cell_coeffs.isel(coeff = (2 * n)).values[0][0])
    
    # Add the harmonic wave component for this order
    y_pred += a_n * np.cos(n * omega * d_plot) + b_n * np.sin(n * omega * d_plot)

    # plot harmonic fit
    plt.plot(cell_data['time'].values, y_pred, color = color, linewidth = 2, label = f'Harmonic Fit (Order {n})')

plt.scatter(cell_data['time'].values, 
            cell_data.isel(lat = 0, lon = 0).values, 
            alpha = 0.3, label = 'Original Data')
plt.xlabel('Day of Year')
plt.ylabel(var)
plt.legend()
plt.title(f'Multi-Order Harmonic Reconstruction (Order {order})')
plt.show()





# %%














