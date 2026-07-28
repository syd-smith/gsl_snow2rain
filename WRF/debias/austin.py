"""
Author: Sydney Smith & Austin LaMontagne
Date Created: July 24, 2026
"""

import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import scipy
import sys
import xarray as xr
import xesmf as xe


# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(current_dir))

# ===========================
# ---- Global Variables ----
# ===========================

# Dictionary that calls name of WRF variable based on the given gridMET variable
WRF_vars = {
    "tmmn": "T2",
    "tmmx": "T2",
    "pr": "RAINNC",
    "sph": "Q2",
    "srad": "SWDOWN",
    "vas": "V10",
    "uas": "U10",
}

# Dictionary that calls name of variable that gridMET data is saved to within the xarray Dataset based on the variable name the file is saved to
MET_vars = {
    "tmmn": "air_temperature",
    "tmmx": "air_temperature",
    "pr": "precipitation_amount",
    "sph": "specific_humidity",
    "srad": "surface_downwelling_shortwave_flux_in_air",
    "vas": "vas",
    "uas": "uas",
}

# =====================
# ---- Functions ----
# =====================

def get_fpaths(f_loc):
    """
    Input path to directory that contains WRF data. Outputs a sorted list of file names.
    """
    # Create list to store file names in
    files = sorted(glob.glob(f'{f_loc}/wrfout_d03*'))

    return files

def loc_mask(ds, dataset_type, lat_min = 34.43, lat_max = 46.57, lon_min = -117.74, lon_max = -100.57):
    """
    Mask given dataset based on predefined latitude and longitude values. Dataset type must be WRF, gridMET, or other.
    Default coordinates are for gridMET interpolation (exculdes locations over the Pacific Ocean).
    """

    if dataset_type == 'WRF':
        # Select gridpoints only within the given spatial boundaries
        mask = (
            (ds['XLAT'] >= lat_min) & (ds['XLAT'] <= lat_max) &
            (ds['XLONG'] >= lon_min) & (ds['XLONG'] <= lon_max)
        )

        # Apply mask to ds
        ds_masked = ds.where(mask, drop = True)

    elif dataset_type == 'gridMET':
        # Select gridpoints only within the given spatial boundaries
        ds_masked = ds.sel(
            lat = slice(lat_max, lat_min), 
            lon = slice(lon_min, lon_max)
        )
    
    elif dataset_type == 'other':
        # Select gridpoints only within the given spatial boundaries
        mask = (
            (ds['lat'] >= lat_min) & (ds['lat'] <= lat_max) &
            (ds['lon'] >= lon_min) & (ds['lon'] <= lon_max)
        )

        # Apply mask to ds
        ds_masked = ds.where(mask, drop = True)

    else:
        # todo: log error here
        print('Select valid dataset type.')
        ds_masked = None

    return ds_masked

def interpo_MET(WRF_file, location, var, year):
    """
    Open one year of gridMET data at a time and interpolate to WRF grid.
    """
    # Open test file and pull data
    WRF_ds = xr.open_dataset(WRF_file)

    # Mask WRF data
    WRF_masked = loc_mask(WRF_ds, 'WRF')

    # Extract lat and long values and squeeze out the time dimension
    lat = WRF_masked['XLAT'].isel(Time = 0).values
    lon = WRF_masked['XLONG'].isel(Time = 0).values

    # Close files to erase from memory
    WRF_ds.close()
    WRF_masked.close()

    # Define file path and open gridMET data for defined variable
    fpath = f'{location}{var}_{year}.nc'
    MET_ds = xr.open_dataset(fpath)[MET_vars[var]] # use MET_vars dictionary to access the variable by the name it's saved to in xarray
    time_vals = MET_ds['day']
    n_times = len(time_vals)

    # Mask gridMET data to size of study region
    MET_masked = loc_mask(MET_ds, 'gridMET')

    # Lat values are ordered from max to min and need to be flipped
    MET_masked = MET_masked.isel(lat = slice(None, None, -1))

    # Keep lat and lon dims as 2d for regridding process
    dims_2d = ('south_north', 'east_west')

    # Create a blueprint for a new dataset to house the interpolated data
    ds_map = xr.Dataset(
        {
            var: (['time', 'lat', 'lon'], MET_masked.values)
        },
        coords = {
            'time': ('time', time_vals.data),
            'lat': (dims_2d, lat),
            'lon': (dims_2d, lon),
        }
    )

    # Make a regridder to apply to gridMET data
    regridder = xe.Regridder(MET_masked, ds_map, method = 'bilinear', extrap_method = 'inverse_dist')

    # House interpolated DataArray in RAM not on disk
    da_regridded = regridder(MET_masked)

    # Lat and lon dims really should be 3D to match WRF
    dims_3d = ('time', 'south_north', 'east_west')

    # Repeat 2D spatial grid across all time steps
    lat_3d = np.repeat(np.expand_dims(lat, axis = 0), n_times, axis = 0)
    lon_3d = np.repeat(np.expand_dims(lon, axis = 0), n_times, axis = 0)

    # Turn DataArray into Dataset
    ds_out = xr.Dataset(
        {
            var: (['time', 'lat', 'lon'], da_regridded.values)
        },
        coords = {
            'time': ('time', time_vals.data),
            'lat': (dims_3d, lat_3d),
            'lon': (dims_3d, lon_3d),
        }
    )

    # Close unnecessary files out of memory
    MET_ds.close()
    ds_map.close()
    MET_masked.close()

    # TODO: make sure we're not losing anything on the edges that takes off the GSL region
    return ds_out

def WRF_daily(date, files, var):
    """
    Calculate daily average WRF value for given variable. Daily data is saved as netCDF to output_dir.
    """

    # List of files for given date
    matched_files = [f for f in files if f'wrfout_d03_{date}' in f]

    # Empty list to fill with daily data
    temp_clean = []

    for file in matched_files:
        # Open one timestamp file at a time
        with xr.open_dataset(file) as ds:
            
            # Only pull out data for given variable
            var_data = ds[WRF_vars[var]]
            xtime_vals = var_data['XTIME'].values
            lat_vals = ds['XLAT'].values
            lon_vals = ds['XLONG'].values

            # Recognize that dims should be 2D
            dims_3d = ('time', 'south_north', 'east_west')

            # Create new dataset to save data to
            clean_ds = xr.Dataset(
                {
                    var: (['time', 'lat', 'lon'], var_data.data)
                },
                coords = {
                    'time': ('time', xtime_vals),
                    'lat': (dims_3d, lat_vals),
                    'lon': (dims_3d, lon_vals)
                }
            )
            
            # Save each timestamp worth of day to predefined list
            temp_clean.append(clean_ds)

    # Concatonate four timestamp files into one file
    combo_clean = xr.concat(temp_clean, dim = 'time')
    # TODO: add check that time dimension only has 4 time stamps

    if var == 'tmmx':
        # Select daily max along XTIME dim for every gridpoint
        adj_data = combo_clean[var].max(dim = 'time')

        # Expand time dim back out after it was collapsed
        daily_data = adj_data.expand_dims('time')

    elif var == 'tmmn':
        # Select daily min along XTIME dim for every gridpoint
        adj_data = combo_clean[var].min(dim = 'time')
        # TODO: check if this is most efficient way because it seems to be slower

        # Expand time dim back out after it was collapsed
        daily_data = adj_data.expand_dims('time')

    elif var == 'pr':
        # WRF stores precipitation as a continuously increasing staircase - difference first and last stairstep in a day to get true precip value for that day
        adj_data= combo_clean.isel(time = 0) - combo_clean.isel(time = 3)
        daily_data = adj_data[var].expand_dims('time')

    elif var == 'sph':
        # Find daily average 
        daily_avg = combo_cleam[var].mean(dim = 'time')
        
        # Convert Q2 to specific humidity
        adj_data = daily_avg / (1 + daily_avg)

        # Expand time dim back out after it was collapsed
        daily_data = adj_data.expand_dims('time')

    else:
        # Calculate daily average
        daily_adj = combo_clean[var].mean(dim = 'time')

        # Expand time dim back out after it was collapsed
        daily_data = adj_data.expand_dims('time')
    
    # TODO: ensure daily_data is array not dataset

    # Save daily data to a new dataset
    daily_ds = xr.Dataset(
        {
            var: (['time', 'lat', 'lon'], daily_data.values)
        },
        coords = {
            'time': ('time', [date]),
            'lat': (dims_3d, lat_vals),
            'lon': (dims_3d, lon_vals)
        }
    )

    # TODO: make relative path
    # Create output directory to store new cleaned files
    output_dir = current_dir / 'daily' / var 
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Save to netcdf
    out_path = os.path.join(output_dir, f'daily_{var}_{date}.nc')
    daily_ds.to_netcdf(out_path)
    print(f'File saved to: {out_path}')

    return daily_ds

def bias_daily(WRF_data, MET_data, date, var):
    """
    Calculate the daily bias (WRF - gridMET) and save data to netCDF.
    """
    # Calculate bias
    if var == 'pr':
        daily_bias = WRF_data / MET_data
    else:
        daily_bias = WRF_data - MET_data

    # TODO: make relative path
    # Create output directory to store new cleaned files
    output_dir = current_dir / 'bias' / var
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Save to netcdf
    out_path = os.path.join(output_dir, f'bias_{var}_{date}.nc')
    daily_bias.to_netcdf(out_path)
    print(f'File saved to: {out_path}')

    return daily_bias

def climate_avg(var):
    """
    Calculate the climatological average (across historical period) bias for a given day.
    """
    # TODO: make relative path
    # Location of bias data from bias_daily
    bias_data = current_dir / 'bias' / var

    # Create output directory to store new cleaned files
    output_dir = current_dir / 'climate_avg' / var
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # TODO: set to correct date range (full year)
    # Generate normal date range with a dummy year (note 1985 is not a leap year)
    dates = pd.date_range(start = '1985-01-01', end = '1985-03-01', freq = 'D')

    # Strip out the year and keep only month-day
    month_days = dates.strftime('%m-%d')

    for day in month_days:
        list_of_files = sorted(glob.glob(f'{bias_data}/bias_{var}_*{day}.nc'))
        # TODO: create error log if list is empty
        all_years = xr.open_mfdataset(list_of_files, combine = 'nested', concat_dim = 'time')

        # Take the mean of all years in the historical period
        climate_avg_of_day = all_years.mean(dim = 'time').load() # computes mean into RAM and detaches it from files
        # print(climate_avg_of_day)

        # Mannual add back time dimension (1985 is just a placeholder year)
        date_add_back = climate_avg_of_day.assign_coords(time = pd.to_datetime(f'1985-{day}'))
        # print(date_add_back)
        # print(date_add_back['time'].values)

        # Reconstruct proper lat and lon dimensions
        lat_2d = all_years['lat'].isel(time = 0)
        lon_2d = all_years['lon'].isel(time = 0)
        date_add_back['lat'] = lat_2d
        date_add_back['lon'] = lon_2d   

        # Save to netcdf
        out_path = os.path.join(output_dir, f'climate_avg_{var}_{day}.nc')
        date_add_back.to_netcdf(out_path)
        print(f'File saved to: {out_path}')

        # Close files out of memory
        all_years.close()
        climate_avg_of_day.close()
        date_add_back.close()

def harmonic_n(t, a0, *coeffs):
    """
    Harmonic function custom to given order (based on number of coefficents passed in *coeffs). 
    This function is used like a calculator for scipy (via curvefit) to find the curve that is
    matches the noisy data the best. 
    """
    # *coeffs: captures any number of coefficients (a1, b1, a2, b2, a3, b3...)
    # omega: fundamental frequency of the entire dataset window (a year worth of daily data)
    omega = (2*np.pi)/365
    # TODO: 
    # TODO: error if t isnt measured in days because then theres a mismatch with omega

    res = a0
    for i in range(0, len(coeffs), 2):
        n = (i // 2) + 1
        res += coeffs[i] * np.cos(n * omega * t) + coeffs[i+1] * np.sin(n * omega * t)
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

# TODO: pr doesnt like curve fit because it has a bunch of nans and inf (maybe from calculating ratio rather than difference for bias)
# create harmonics directory to put dataset in and keep graphs of what functions are best
# try using xr.curvefit to wrap harmonic function by location
# test what order is the best fit
# interpolate daily bias based on curve from harmonic function
# extract bias from daily WRF data and save
# TODO: make sure that harmonic function 





def main(var, WRF_in, MET_in):
    
    # TODO: log errors if the workflow isn't completed sequentially
    # Generate list of WRF input files
    files = get_fpaths(WRF_in)
    
    # TODO: set to full historical period
    for year in range(1985, 1987):
        # Call and interpolate gridMET data for the given year 
        MET_data = interpo_MET(files[0], MET_in, var, year) # pass first file in files as example grid

        # Create date range using pandas
        # TODO: set to dates for full year
        dates = pd.date_range(start = f'{year}-01-01', end = f'{year}-03-01', freq = 'D') 

        for day in dates:
            # Turn day in to usable date string 
            day_str = day.strftime('%Y-%m-%d')

            # Create clean file of daily WRF data
            daily_avg = WRF_daily(day_str, files, var)

            # Select day worth of gridMET data
            MET_select = MET_data.sel(time = day_str)

            # Find bias for given data
            bias = bias_daily(daily_avg, MET_select, day_str, var)

            # Close daily files out of memory
            daily_avg.close()
            bias.close()

        # Close out of gridMET data once the entire year is complete
        MET_data.close()
        print(f'{year} daily bias caclulations complete!')
    
    # Calculate the climatological daily bias
    climate_avg(var)

    # Calculate fourier coefficients using custom harmonic function
    # Outputs saved the harmonics directory
    for order in range(1, 4):
        apply_harmonic(var, order)

    # Pass coefficients back into custom harmonic function to get smoothed data


# ======================
# ---- Entry Point ----
# ======================

# TODO: explain entry point and what variables need to be defined
# Set variable based on gridMET variable save names
# tmmn, tmmx, pr, sph, srad, vas, uas
# TODO: check for edge cases with other variables 

if __name__ == '__main__':
    main(
        var = 'tmmn',
        WRF_in = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/', 
        MET_in = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/'
        )

