"""
Author: Sydney Smith
Date Created: August 17, 2026
"""

from datetime import datetime
import glob
from ibicus.debias import ECDFM
from loguru import logger
import metpy.calc as mpcalc
import numpy as np
import os
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import sys
import time
import xarray as xr

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from old.temporal_chunks import get_fpaths, interpo_MET, WRF_daily

# ===================
# - Set Up Logger - 
# ==================

# Create directory for log files if it doesn't already exist
log_path = str(current_dir / 'log')
os.makedirs(log_path, exist_ok = True)

# String filename
log_filename = datetime.now().strftime('log/%Y-%m-%d_%H-%M-%S.log')

# Custom format for log prints
log_format = log_format = '<cyan>{time:HH:mm:ss}</cyan> | <level>{level:>8}</level> | <yellow>{name}</yellow>:<cyan>{function}</cyan>:line <magenta>{line}</magenta> - <level>{message}</level>'

# Removes the default stderr sink
logger.remove()  
# Anything above info to console
logger.add(sys.stderr, colorize = True, format = log_format, level = 'INFO') 
# Anything above debug to log file
logger.add(log_filename, colorize = False, format = log_format, level = 'DEBUG')
logger.debug(f'Log files saved to {log_path}')

# ===========================
# ---- Global Variables ----
# ===========================

# Dictionary that calls name of WRF variable based on the given gridMET variable
bias_vars = {
    'tmmn': 'tasmin',
    'tmmx': 'tasmax',
    'pr': 'pr',
    'sph': 'sph',
    'srad': 'rsds',
    'vas': 'sfcwind',
    'uas': 'sfcwind',
}

# =====================
# ---- Functions ----
# =====================

def fix_time_coord(ds):

    # Ensure the time coordinate is always parsed as a proper datetime object
    if 'time' in ds.coords:
        ds['time'] = pd.to_datetime(ds['time'].values)

    return ds

def data_saver(data, destination, var, year):
    """
    Save gridMET data to a netcdf file.
    """
    # Create output directory to store new cleaned files
    output_dir = current_dir / destination / var
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Save to netcdf
    out_path = os.path.join(output_dir, f'{var}_{year}.nc')
    data.to_netcdf(out_path)
    logger.success(f'File saved to: {out_path}')

    # Close out of data once saved
    data.close()
    logger.success(f'{var}-{year} data saved!')

def convert_pr(ds, output_units):
    """
    Convert precipitation flux (kg m-2 s-1) into precipitation depth (mm/day) or vice versa.
    1mm of water equals 1kg m-2 of water.
    """
    if output_units == 'mm/day':
        # Convert flux to depth
        conversion = ds * 86400

    elif output_units == 'kg m-2 s-1':
        # Convert depth to flux
        conversion = ds / 86400
    
    else:
        logger.error(f'{output_units} is an invalid output for this function. Units must be mm/day or kg m-2 s-1.')

    logger.success('Precipitation conversion success!')
    return conversion

def convert_wind(u, v, speed):
    """
    Convert debiased wind magnitude outputs into u and v components of wind. 
    """

    alpha = mpcalc.wind_direction(u, v)
    u, v = mpcalc.wind_components(speed, alpha)

    return u, v

def apply_debiaser(var, obs, hist, fut):
    """
    Debias WRF data using ECDFM method from ibicus. This method preserves the trend in all quantiles. 
    Parameters are set to match quantile mapping debiasing used in the MACA downscaling process (GSLBIP). 
    See documentation for more information on the code libaray used.
    https://ibicus.readthedocs.io/en/latest/reference/debias.html#ibicus.debias.ECDFM
    """
    # TODO: Verify that empirical distribution setup is proper method in this use case
    # Set up RV histogram distribution from obs to simulate empirical dsitribution of the data
    hist_vals, bin_edges = np.histogram(obs, bins = 'auto', density = True)
    empirical_dist = stats.rv_histogram((hist_vals, bin_edges))

    # Check for zeros in the datasets
    obs_zeros = (obs == 0).any()
    logger.info(f'Obs dataset has zeros at: {obs_zeros}')
    hist_zeros = (hist == 0).any()
    logger.info(f'Historical dataset has zeros at: {hist_zeros}')
    fut_zeros = (fut == 0).any()
    logger.info(f'Future dataset has zeros at: {fut_zeros}')

    if var == 'pr':
        # Convert precipitation from a depth to a flux
        obs = convert_pr(obs, 'kg m-2 s-1')
        hist = convert_pr(hist, 'kg m-2 s-1')
        fut = convert_pr(fut, 'kg m-2 s-1')

        # TODO: Check that precipitation should be nonparametric
        # Expected units: kg m-2 s-1
        debiaser = ECDFM.for_precipitation(
            model_type = 'hurdle',  # 
            # amounts_distribution = None,  # Avoids parametric fitting like Gamma
            censor_values_to_zero = True, # Set values below sensoring threshold to zero
            censoring_threshold = 1.1574074e-06, # Equal to 0.1 mm/day
            running_window_mode = True, 
            running_window_length = 31, 
            running_window_step_length = 1 # Step size of one day for daily data
        )

    elif var in ['tmmn', 'tmmx', 'wind']: 
        # Temp expected units: K
        # Wind expected units: m s-1
        debiaser = ECDFM.from_variable(
            bias_vars[var], 
            running_window_mode = True,
            running_window_length = 31, 
            running_window_step_length = 1
        )

    else:
        logger.info(f'Using custom debiaser setup to debias {var} data!')

        # Intended for sph and srad
        # Instantiate as unbounded variables with custom settings
        debiaser = ECDFM(
            distribution = empirical_dist, 
            cdf_threshold = 0.0,  # Ensures lower bound handling at zero 
            running_window_mode = True,
            running_window_length = 31, 
            running_window_step_length = 1
        )

    # Apply debiaser to data and log success
    debiased_data = debiaser.apply(obs, hist, fut) # Runs data in parallel under the hood using the predefined dask chunks
    logger.success('Quantile mapping debiaser applied to data.')

    return debiased_data

def debiaser_setup(var): 
    """
    Setup debiaser using xr.apply_ufunc to process each location along a timeseries. This 
    requires lazy loading all of the data first before applying the debiaser.
    """
    # Define paths 
    obs_path = current_dir / 'gridMET'
    model_path = current_dir / 'daily'

    if var == 'wind':
        # Open datasets lazily to not overload memory
        u_obs = xr.open_dataset(next(obs_path.glob(f'*{var}*.nc')), decode_times = True)
        u_model = xr.open_dataset(next(model_path.glob(f'*{var}*.nc')), decode_times = True)

        # Open datasets lazily to not overload memory
        v_obs = xr.open_dataset(next(obs_path.glob(f'*{var}*.nc')), decode_times = True)
        v_model = xr.open_dataset(next(model_path.glob(f'*{var}*.nc')), decode_times = True)
        
        # Combine u and v components into magnitude
        # TODO: If mpcalc throws errors because of dask chunking just perform calculations using simple python operations
        obs = mpcalc.wind_speed(u_obs, v_obs)
        model = mpcalc.wind_speed(u_model, v_model)
        
        # Isolate model historical period
        hist = model.sel(time = slice('1985-01-01', '2014-12-31'))
  
    else:
        # Open datasets lazily to not overload memory
        obs = xr.open_dataset(next(obs_path.glob(f'*{var}*.nc')), decode_times = True)
        model = xr.open_dataset(next(model_path.glob(f'*{var}*.nc')), decode_times = True)
        
        # Isolate model istorical period
        hist = model.sel(time = slice('1985-01-01', '2014-12-31'))

    # Log success of data load
    logger.success('Lazy loaded all datasets for debiasing.')

    # Extract data from xr.dataset as numpy arrays
    obs_vals = obs[var].values
    hist_vals = hist[var].values
    model_vals = model[var].values

    # Apply debiaser at all locations (location handling is done by ibicus library)
    debiased = apply_debiaser(var, obs_vals, hist_vals, model_vals)
    logger.info(debiased)

    # Consider apply_ufunc only if built in location looping seems unable to handle the size of the dataset
    logger.info('Exited out of apply_debiaser. On to saving data!')

    # Ensure that the debiased data is the same shape as the future dataset
    assert debiased.shape == model[var].shape or logger.error('Debiased dataset is not the same shape as the model dataset for xarray reconstruction.')
    
    # Reconstruct xr.dataset using model's metadata
    # Build out fut and hist datasets separately first
    ds_debiased = model.copy(deep = True)
    ds_debiased[var].values = debiased
    logger.info(ds_debiased)
    logger.success('Dataset reconstructed!')
    
    # Convert debiased data to datetime object
    ds_debiased['time'] = pd.to_datetime(ds_debiased['time'].values)

    # Create output directory to store new cleaned files
    output_dir = current_dir / 'wrfout' 
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    if var == 'pr':
        # Convert precipitation back to depth
        ds_debiased = convert_pr(ds_debiased, 'mm/day')

        # Generate save name for debiased data
        out_path = os.path.join(output_dir, f'wrfout_GSLBIP_multimodel_ssp245_{var}.nc')

        # Save data to netCDF file
        ds_debiased.to_netcdf(out_path)
        logger.success(f'File saved to: {out_path}')

        # Close out of data once saved
        ds_debiased.close()

    elif var == 'wind':
        # TODO: fix memories issues from mpcalc and wind conversion
        # Convert wind magnitude back to u and v
        results = convert_wind(u_model, v_model, ds_debiased)
        variables = ['uas', 'vas']
        # TODO: Check the variable names in results after splitting wind back into uas and vas

        # Save both wind component separately
        for variable, result in zip(variables, results):
            # Generate save name for debiased data
            out_path = os.path.join(output_dir, f'wrfout_GSLBIP_multimodel_ssp245_{variable}.nc')

            # Save data to netCDF file
            result.to_netcdf(out_path)
            logger.success(f'File saved to: {out_path}')

            # Close out of data once saved
            result.close()

    else:
        # Generate save name for debiased data
        out_path = os.path.join(output_dir, f'wrfout_GSLBIP_multimodel_ssp245_{var}.nc')

        # Save data to netCDF file
        ds_debiased.to_netcdf(out_path)
        logger.success(f'File saved to: {out_path}')

        # Close out of data once saved
        ds_debiased.close()

    # Log when complete
    logger.success(f'Bias calculations complete for {var}!')
        
# Catch silent errors and report to log file
@logger.catch 
def main(variable, domain, WRF_in, MET_in, debias = True):
    # State what variable is being used
    logger.info(f'Debiasing for {variable}.')

    if variable == 'wind':
        variables = ['uas', 'vas']
    else:
        variables = [variable]

    # for var in variables:
    #     # TODO: log errors if the workflow isn't completed sequentially
    #     # Generate list of WRF input files
    #     files = get_fpaths(WRF_in, domain)
        
    #     # Set to full historical period
    #     for year in range(1985, 2015):
    #         # Call and interpolate gridMET data (obs) for the given year 
    #         MET_data = interpo_MET(files[0], MET_in, var, year) # pass first WRF file in files as example grid

    #         # Save year of interpolated gridMET data
    #         save_data = data_saver(MET_data, 'gridMET', var, year)

    #     # Combine all gridMET files into a single file with context manager
    #     with xr.open_mfdataset(
    #         glob.glob(str(current_dir / 'gridMET' / var / '*.nc')), 
    #         combine = 'nested', 
    #         concat_dim = 'time', 
    #         preprocess = fix_time_coord
    #         ) as gridmet_open:

    #         # Write file to netcdf
    #         gridmet_open.sortby('time').to_netcdf(current_dir / 'gridMET' / f'gridMET_GSLBIP_{var}.nc')
        
    #     logger.success(f'All gridMET files successfully interpolated and saved for {var}!')

    #     # Set to historical + future period
    #     for year in range(1985, 2100):
    #         # Create date range using pandas
    #         # TODO: set to dates for full year
    #         dates = pd.date_range(start = f'{year}-01-01', end = f'{year}-12-31', freq = 'D') 

    #         for day in dates:
    #             if day == dates[0]:
    #                 # Set the day you want to be working with to today
    #                 today = day.strftime('%Y-%m-%d') # Turn day in to usable date string 
    #                 continue

    #             else:
    #                 # Because of the offset from UTC to MT you need to pull in the next day worth of data as well
    #                 tomorrow = day.strftime('%Y-%m-%d')

    #                 # Create clean file of daily WRF data
    #                 daily_avg = WRF_daily(today, tomorrow, files, var, domain, current_dir)

    #                 # Set tomorrow as the new today to move on to the next series
    #                 today = tomorrow
        
    #     # Combine all cleaned WRF files into a single file using context manager
    #     with xr.open_mfdataset(
    #         glob.glob(str(current_dir / 'daily' / var / '*.nc')), 
    #         combine = 'nested', 
    #         concat_dim = 'time', 
    #         preprocess = fix_time_coord
    #         ) as wrf_open:

    #         # Write file to netcdf
    #         wrf_open.sortby('time').to_netcdf(current_dir / 'daily' / f'wrf_raw_GSLBIP_multimodel_ssp245_{var}.nc')

    #     logger.success(f'WRF files successfully cleaned and saved for {var}!')

    if debias:
        # Apply debiaser to data
        debiased_fut = debiaser_setup(variable)

# ======================
# ---- Entry Point ----
# ======================

# TODO: explain entry point and what variables need to be defined
# Set variable based on gridMET variable save names
# tmmn, tmmx, pr, sph, srad, wind (vas & uas)
# TODO: check for edge cases with other variables 

if __name__ == '__main__':
    # Track program time in log files
    start = time.perf_counter()
    logger.info('Beginning execution.')
    logger.info('No chunker.')

    # Only inputs required
    main(
        variable = 'tmmx',
        domain = '03',
        WRF_in = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/', 
        MET_in = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/',
        debias = True
        )

    # Report of runtime at completion 
    logger.success(f'Debiasing process completed!')
    logger.info(f'Total runtime: {time.perf_counter() - start:.4f}s')

    # Force script to stop running once code is finished
    sys.exit(0)

