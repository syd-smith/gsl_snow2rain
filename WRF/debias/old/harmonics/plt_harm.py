# %%
"""
Author: Sydney Smith
Date Created: August 17, 2026
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
import xarray as xr

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from temporal_chunks import loc_mask

def harmonic_n(d, a0, *coeffs):
    """
    Harmonic function custom to given order (based on number of coefficents passed in *coeffs). 
    This function is used like a calculator for scipy (via curvefit) to find the curve that is
    matches the noisy data the best. 
    """
    # *coeffs: captures any number of coefficients (a1, b1, a2, b2, a3, b3...)
    # omega: fundamental frequency of the entire dataset window (a year worth of daily data)
    omega = (2*np.pi)/365
    # TODO: error if d isn't measured in days because then theres a mismatch with omega

    res = a0
    for i in range(0, len(coeffs), 2):
        n = (i // 2) + 1
        res += coeffs[i] * np.cos(n * omega * d) + coeffs[i+1] * np.sin(n * omega * d)
    return res

    # TODO: check what time values are set to in the dataset (should be numerical offset relative to start date of the dataset)

def order_coeffs(order):
    """
    Define the coefficients that should be used for the given order of harmonic function. Value passed for order must be an integer. 
    """

    # Create list to store values in (all lists should start with the initial guess)
    coeffs = ['a0']

    # Loop through coefficients to add correct number based on predefined order
    for step in range(1, order+1):
        coeffs.append(f'a{step}')
        coeffs.append(f'b{step}')
    # TODO: add to log
    # print(coeffs)

    return coeffs

def apply_harmonic(var, order):
    """
    Apply custom harmonic function to noisy data. Value passed for order must be an integer. 
    """
    # Load entire year worth of climatological daily bias
    climate_dir = current_dir / 'climate_avg' / var
    climate_files = sorted(glob.glob(f'{climate_dir}/climate_avg_{var}*.nc'))
    climate_data = xr.open_mfdataset(climate_files, combine = 'nested', concat_dim = 'time').load()

    # Define coefficents for given order of harmonic function
    order_params = order_coeffs(order)

    # Create dict of initial guesses
    initial_guesses = {'a0': climate_data[var].mean('time')}

    # Set inital guess of coeffs to 0 to prevent convergence failures
    for param in order_params:
        if param != 'a0':
            initial_guesses[param] = 0

    # Apply harmonic function to entire xarray.Dataset (unique function for each location)
    fit_results = climate_data.curvefit(
        coords = 'time', # tells harmonic_n to use 'time' for t
        func = harmonic_n,
        param_names = order_params,
        # a0: offset or essentially what the mean of the data is (inital guess)
        p0 = initial_guesses,
        errors = 'ignore' # sets errors to NAN instead of breaking the script
    )

    dim_mapping = {}
    if 'lat' in fit_results.dims:
        dim_mapping['lat'] = 'south_north'
    if 'lon' in fit_results.dims:
        dim_mapping['lon'] = 'east_west'
        
    if dim_mapping:
        fit_results = fit_results.rename_dims(dim_mapping)

    # 2. Extract clean 2D lat and lon grids and assign them as coordinates mapped to south_north/east_west
    lat_2d = climate_data['lat'].isel(time = 0, drop = True) if 'time' in climate_data['lat'].dims else climate_data['lat']
    lon_2d = climate_data['lon'].isel(time = 0, drop = True) if 'time' in climate_data['lon'].dims else climate_data['lon']

    fit_results = fit_results.assign_coords({
        'lat': (('south_north', 'east_west'), lat_2d.values),
        'lon': (('south_north', 'east_west'), lon_2d.values)
    })

    # Create output directory to store harmonics data
    output_dir = current_dir / 'harmonics'
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists
    
    # Save data to netCDF
    out_path = os.path.join(output_dir, f'harmonic{order}_{var}.nc')
    fit_results.to_netcdf(out_path)
    print(f'File saved to: {out_path}')

    fit_results.close()

def order_call(ds, order, var):
    """
    Unpacks necessary coefficients from curvefit dataset.
    """
    # Access offset coeff (in every dataset)
    a0 = ds[f'{var}_curvefit_coefficients'].sel(param = 'a0').values.item()
    
    # Create list to store coeffs in
    coeff = [a0]

    # Loop through however many coeff are in that order of harmonic function
    for i in range(1, order+1):
        coeff.append(ds[f'{var}_curvefit_coefficients'].sel(param = f'a{i}').values.item())
        coeff.append(ds[f'{var}_curvefit_coefficients'].sel(param = f'b{i}').values.item())

    return coeff

def harmonic_plt(noise, first, second, third, var):
    """
    Plot scatter plot of noise data and fits of harmonic functions.
    """

    # Initialize plot
    fig, ax = plt.subplots(figsize = (12, 6))

    # Also ensure your noise variable is 1D for that specific masked point
    noise_y = noise[var].isel(lat = 0, lon = 0).squeeze()

    # Noise scatter plot
    ax.plot(
        noise['time'], 
        noise_y, 
        'ko', 
        alpha = 0.4, 
        markersize = 4, 
        label = 'Noisy Data (Climatology)'
    )

    # First order harmonic fit
    ax.plot(
        noise['time'], 
        first, 
        'r-', 
        linewidth = 2, 
        label = '1st Order Harmonic'
    )

    # Second order harmonic fit
    ax.plot(
        noise['time'], 
        second, 
        'g-', 
        linewidth = 2, 
        label = '2nd Order Harmonic'
    )

    # Third order harmonic fit
    ax.plot(
        noise['time'], 
        third, 
        'b-', 
        linewidth = 2, 
        label = '3rd Order Harmonic'
    )

    ax.set_title(f'{var} Climatology', fontsize = 14)
    ax.set_xlabel('Time', fontsize = 12)
    ax.set_ylabel(var, fontsize = 12)
    ax.grid(True, linestyle = '--', alpha = 0.5)
    ax.legend(frameon = True)

    plt.tight_layout()
    plt.show()

def main():

    # Define variable of interest
    var = 'tmmn'

    # Load entire year worth of climatological daily bias data (raw noisy datapoints)
    climate_dir = parent_dir / 'climate_avg' / var
    climate_files = sorted(glob.glob(f'{str(climate_dir)}/climate_avg_{var}*.nc'))
    noise = xr.open_mfdataset(climate_files, combine = 'nested', concat_dim = 'time').load() # Load noise into RAM and strip from other files to save space

    # Load results from harmonic function calculations
    dir = '/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/harmonics/'
    first = xr.open_dataset(f'{dir}harmonic1_{var}.nc')
    second = xr.open_dataset(f'{dir}harmonic2_{var}.nc')
    third = xr.open_dataset(f'{dir}harmonic3_{var}.nc')

    # Set exmaple lat and lon values
    lat_min, lat_max = 40.12, 40.13
    lon_min, lon_max = -110.09, -110.04
    time_dim = noise['time'].values.astype('float64')

    # Mask data to location of choice
    noise_masked = loc_mask(noise, 'other', lat_min, lat_max, lon_min, lon_max)
    first_masked = loc_mask(first, 'other', lat_min, lat_max, lon_min, lon_max)
    second_masked = loc_mask(second, 'other', lat_min, lat_max, lon_min, lon_max)
    third_masked = loc_mask(third, 'other', lat_min, lat_max, lon_min, lon_max)

    # Get coefficients for that order
    first_coeff = order_call(first_masked, 1, var)
    second_coeff = order_call(second_masked, 2, var)
    third_coeff = order_call(third_masked, 3, var)

    # Pass them into your harmonic function to generate the smooth curve
    # * tells it to unpack the list
    smooth_first = harmonic_n(time_dim, *first_coeff) 
    smooth_second = harmonic_n(time_dim, *second_coeff)
    smooth_third = harmonic_n(time_dim, *third_coeff)

    # Squeeze out any extra dimensions (like lat, lon, or singleton axes) so it becomes a 1D array of length 60
    smooth_first_1d = np.squeeze(smooth_first)
    smooth_second_1d = np.squeeze(smooth_second)
    smooth_third_1d = np.squeeze(smooth_third)

    # Call plotting function
    plot_harms = harmonic_plt(noise_masked, smooth_first_1d, smooth_second_1d, smooth_third_1d, var)

if __name__ == '__main__':
    main()






# %%
