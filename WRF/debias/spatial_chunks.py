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

def debias(var):

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

    var = 'pr'
    # Define paths for observation and model data
    obs_path = current_dir / 'gridMET' / var
    print(obs_path)
    model_path = current_dir / 'daily' / var
    print(model_path)

    # Open datasets lazily using dask chunks
    obs = xr.open_mfdataset(glob.glob(str(obs_path / '*')), chunks = {'lat': 10, 'lon': 10})
    print(obs)
    # hist = xr.open_mfdataset(model_path, chunks = {'lat': 10, 'lon': 10})
    # future_ds = xr.open_dataset("cm_future_precipitation.nc", chunks = {'lat': 10, 'lon': 10})



# Catch silent errors and report to log file
@logger.catch 
def main(var, domain, WRF_in, MET_in):
    # State what variable is being used
    logger.info(f'Debiasing for {var}.')

     # TODO: log errors if the workflow isn't completed sequentially
    # Generate list of WRF input files
    files = get_fpaths(WRF_in, domain)
    
    # TODO: set to full historical period
    for year in range(1985, 2015):
        # Call and interpolate gridMET data for the given year 
        MET_data = interpo_MET(files[0], MET_in, var, year) # pass first file in files as example grid

        # Create date range using pandas
        # TODO: set to dates for full year
        dates = pd.date_range(start = f'{year}-01-01', end = f'{year}-12-31', freq = 'D') 

        for day in dates:
            # Turn day in to usable date string 
            day_str = day.strftime('%Y-%m-%d')

            # Create clean file of daily WRF data
            daily_avg = WRF_daily(day_str, files, var, domain)

            # Close daily files out of memory
            daily_avg.close()

        # Create output directory to store new cleaned files
        output_dir = current_dir / 'gridMET' / var
        os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

        # Save to netcdf
        out_path = os.path.join(output_dir, f'gridMET_{var}_{year}.nc')
        MET_data.to_netcdf(out_path)
        logger.success(f'File saved to: {out_path}')

        # Close out of gridMET data once the entire year is complete
        MET_data.close()
        logger.success(f'{year} daily caclulations complete!')
    
    test = debias(var)
    


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
    logger.info(f'Total runtime: {time.perf_counter() - start:.4f}s')

