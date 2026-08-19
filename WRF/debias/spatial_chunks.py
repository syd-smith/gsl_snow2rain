"""
Author: Sydney Smith
Date Created: August 17, 2026
"""

from datetime import datetime
import glob
from ibicus.debias import ECDFM
from loguru import logger
import os
import pandas as pd
from pathlib import Path
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
WRF_vars = {
    'tmmn': 'T2',
    'tmmx': 'T2',
    'pr': 'RAINNC',
    'sph': 'Q2',
    'srad': 'SWDOWN',
    'vas': 'V10',
    'uas': 'U10',
}

# Dictionary that calls name of variable that gridMET data is saved to within the xarray Dataset based on the variable name the file is saved to
MET_vars = {
    'tmmn': 'air_temperature',
    'tmmx': 'air_temperature',
    'pr': 'precipitation_amount',
    'sph': 'specific_humidity',
    'srad': 'surface_downwelling_shortwave_flux_in_air',
    'vas': 'vas',
    'uas': 'uas',
}

# Dictionary that calls name of WRF variable based on the given gridMET variable
bias_vars = {
    'tmmn': 'tasmin',
    'tmmx': 'tasmax',
    'pr': 'RAINNC',
    'sph': 'Q2',
    'srad': 'rsds',
    'vas': 'V10',
    'uas': 'U10',
}

# =====================
# ---- Functions ----
# =====================

def save_MET(data, var, year):
    """
    Save gridMET data to a netcdf file.
    """
    # Create output directory to store new cleaned files
    output_dir = current_dir / 'gridMET' / var
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Save to netcdf
    out_path = os.path.join(output_dir, f'gridMET_{var}_{year}.nc')
    data.to_netcdf(out_path)
    logger.success(f'File saved to: {out_path}')

    # Close out of gridMET data once the entire year is complete
    data.close()
    logger.success(f'{year} daily caclulations complete!')

def apply_debiaser(var, obs, hist):
    """
    Debias WRF data using ECDFM method from ibicus. Parameters are set to match quantile mapping
    debiasing used in the MACA downscaling process (GSLBIP). See documentation for more information 
    on the code libaray used.
    https://ibicus.readthedocs.io/en/latest/reference/debias.html#ibicus.debias.ECDFM
    """
    if var == 'pr':
        debiaser = ECDFM.for_precipitation(
            model_type = 'hurdle',  # or 'hurdle' depending on how it handles the binary split
            amounts_distribution = None,  # Avoids parametric fitting like Gamma
            censor_values_to_zero = True, # Set values below sensoring threshold to zero
            censoring_threshold = 0.1, # In mm/s (equal to 0.1mm/day)
            distribution = None, # Use empirical distribution
            running_window_length = 31, 
            running_window_step_length = 1 # Step size of one day for daily data
        )

    else:
        debiaser = ECDFM.from_variable(
            var, 
            distribution = None, # Use empirical distribution
            running_window_length = 31, 
            running_window_step_length = 1
        )

    # Apply debiaser to data and log success
    debiased_data = debiaser.apply(obs, hist, fut)
    logger.succes('Quantile mapping debiaser applied to data.')

    return debiased_data

def debiaser_setup(var): 
    """
    Setup debiaser using xr.apply_ufunc to process each location along a timeseries. This 
    requires lazy loading all of the data first before applying the debiaser.
    """
    # Define paths for observation and model data
    obs_path = current_dir / 'gridMET' / var
    model_path = current_dir / 'daily' / var

    # Open datasets lazily to not overload memory
    obs = xr.open_mfdataset(glob.glob(str(obs_path / '*.nc')), combine = 'nested', concat_dim = 'time').sel(time = slice('1985-01-01', '2001-12-31'))
    hist = xr.open_mfdataset(glob.glob(str(model_path / '*.nc')), combine = 'nested', concat_dim = 'time').sel(time = slice('1985-01-01', '2014-12-31'))
    future_ds = xr.open_dataset(glob.glob(str(model_path / '*.nc')), combine = 'nested', concat_dim = 'time').sel(time = slice('2015-01-01', '2099-12-31'))

    # Apply debiaser at all locations using apply_ufunc
    result = xr.apply_ufunc(
        apply_debiaser, # Function being called
        obs_ds[var], hist_ds[var], fut_ds[var], # Passing function arguements
        input_core_dims = [['time'], ['time'], ['time']], # Treat time as the 1D loop unit
        output_core_dims = [['time']],                   
        vectorize = True,                                # Automatically loops over lat/lon
        dask = 'parallelized',                           # Parallelize over chunks
        output_dtypes = [obs_ds['tmmn'].dtype]           # Ensure output matches input type
        )

    # TODO: assert and log - result are proper shape

    # Create output directory to store debiased data
    output_dir = current_dir / 'debiased'
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Save to netcdf
    out_path = os.path.join(output_dir, f'debised_{var}.nc')
    result.to_netcdf(out_path)
    logger.success(f'File saved to: {out_path}')

    # Close data out of memory
    result.close()
    logger.success(f'Bias caclulations complete!')

    return result

# Catch silent errors and report to log file
@logger.catch 
def main(var, domain, WRF_in, MET_in):
    # State what variable is being used
    logger.info(f'Debiasing for {var}.')

    # TODO: log errors if the workflow isn't completed sequentially
    # Generate list of WRF input files
    files = get_fpaths(WRF_in, domain)
    
    # TODO: set to full historical period
    for year in range(1985, 2099):
        # Call and interpolate gridMET data for the given year 
        MET_data = interpo_MET(files[0], MET_in, var, year) # pass first WRF file in files as example grid

        # Save year of interpolated gridMET data
        save_data = save_MET(MET_data, var, year)

        # Create date range using pandas
        # TODO: set to dates for full year
        dates = pd.date_range(start = f'{year}-01-01', end = f'{year}-12-31', freq = 'D') 

        for day in dates:
            # Turn day in to usable date string 
            day_str = day.strftime('%Y-%m-%d')

            # Create clean file of daily WRF data
            daily_avg = WRF_daily(day_str, files, var, domain)

    # Apply debiaser to data
    debiased_data = debiaser_setup(var)

# ======================
# ---- Entry Point ----
# ======================

# TODO: explain entry point and what variables need to be defined
# Set variable based on gridMET variable save names
# tmmn, tmmx, pr, sph, srad, vas, uas
# TODO: check for edge cases with other variables 

if __name__ == '__main__':
    # Track program time in log files
    start = time.perf_counter()
    logger.info('Beginning execution.')

    # Only inputs required
    main(
        var = 'tmmn',
        domain = '03',
        WRF_in = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/', 
        MET_in = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/'
        )

    # Report of runtime at completion 
    logger.success(f'Debiasing process completed for {var}!')
    logger.info(f'Total runtime: {time.perf_counter() - start:.4f}s')

    # Force script to stop running once code is finished
    sys.exit(0)

