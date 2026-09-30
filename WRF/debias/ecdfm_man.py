"""
Author: Sydney Smith
Date Created: September 29, 2026
"""

from datetime import datetime
import glob
from loguru import logger
import numpy as np
import os
import pandas as pd
from pathlib import Path
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
import sys
import time
import xarray as xr

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from plots.validation import running_window_slice, loc_sel

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

# =====================
# ---- Functions ----
# =====================

def apply_to_date(obs: xr.Dataset, hist, raw, var, date) -> np.ndarray:
    """
    Debiases data for a given date by constructing a CDF using a dataset restricted to be within a 
    running window. Date must be a valid date that exists within the dataset formatted as '%Y-%m-%d' 
    datetime object.
    """
    # TODO: check how running_window_slice handles leap years

    # Slice data into 31 day running windows centered on given date (note: 31 days is defualt setting)
    obs_sliced = running_window_slice(obs, date)
    hist_sliced = running_window_slice(hist, date)
    raw_sliced = running_window_slice(raw, date)
    logger.info('Datasets sliced into running windows.')

    # Create CDFs from sliced datasets
    obs_cdf = ECDF(obs_sliced[var].values.flatten())
    hist_cdf = ECDF(hist_sliced[var].values.flatten())
    raw_cdf = ECDF(raw_sliced[var].values.flatten())

    # Select day of interest and create mask
    day = int(date.strftime('%d'))
    month = int(date.strftime('%m'))
    mask = (raw.time.dt.month == month) & (raw.time.dt.day == day)

    # Skip dates that don't exist
    if not mask.any():
        logger.warning(f'No date found for {date}.')
        return None
    
    # Apply mask to dataset
    raw_date = raw.where(mask, drop = True) # This should select day of interest across all years in ds
    logger.info(f'Dataset sliced down to date of interest: {date}')
    logger.info(raw_date)

    # Use raw_cdf like a function to find all of the percentiles in raw_date
    raw_percentiles = raw_cdf(raw_date[var].values)
    logger.info(f'Percentile values for raw data: {raw_percentiles}')

    # Invert hist_cdf and obs_cdf to take a percentile and output a data point
    inv_obs = monotone_fn_inverter(obs_cdf, obs_cdf.x)
    inv_hist = monotone_fn_inverter(hist_cdf, hist_cdf.x)

    # Find data points for every percentile in raw_percentiles
    obs_vals = inv_obs(raw_percentiles)
    logger.info(f'Obs values for given percentiles: {obs_vals}')
    hist_vals = inv_hist(raw_percentiles)
    logger.info(f'Historical values for given percentiles: {hist_vals}')

    # Perform bias adjustment
    bias = hist_vals - obs_vals
    raw_bias_corrected = raw_date[var].values - bias

    logger.info('Raw data successfully bias corrected!')
    logger.info(raw_bias_corrected)
    logger.info(f'Bias corrected max: {raw_bias_corrected.max()}')
    logger.info(f'Bias corrected min: {raw_bias_corrected.min()}')

    # Close out of unwanted datasets
    obs_sliced.close()
    hist_sliced.close()
    raw_sliced.close()
    raw_date.close()

    return raw_bias_corrected

def ECDFM(obs, raw, var):
    """
    Performs ECDFM quantile mapping debiasing process on given data based on data from a given histroical
    period and corresponding observational data. Raw data should include both historical and future data. 
    """

    # Set location of interest as SLC airport
    lat = 40.788
    lon = -111.978

    # Slice raw data down to historical period
    hist = raw.sel(time = slice('1985-01-01', '2014-12-31'))
    
    # Select specific location of interest (choses nearest possible location)
    obs_loc = loc_sel(obs, lat = lat, lon = lon)
    hist_loc = loc_sel(hist, lat = lat, lon = lon)
    raw_loc = loc_sel(raw, lat = lat, lon = lon)
    logger.info(raw_loc[var])

    # Make copy of raw_loc to plug debiased data into
    framework = raw_loc.copy()

    # Create date range that spans a calendar year
    dates = pd.date_range(start = '1985-01-01', end = '1985-12-31', freq = 'D') 

    for date in dates:
        # Apply bias correction for single date
        logger.info(f'Date type: {type(date)}')
        debiased_doy = apply_to_date(obs_loc, hist_loc, raw_loc, var, date)

        # Pull month and day from date
        day = int(date.strftime('%d'))
        month = int(date.strftime('%m'))

        # Plug in debiased data
        mask = (framework.time.dt.month == month) & (framework.time.dt.day == day)
        framework[var].values[mask] = debiased_doy 
        logger.info(f'Debiased data for {date} added to dataset!')

    logger.info('Raw model data.')
    logger.info(raw_loc[var].values)
    logger.info('Debiased model data.')
    logger.info(framework[var].values)

    return framework

# Catch silent errors and report to log file
@logger.catch 
def main(var, data_location):
    # Open datasets
    obs_path = glob.glob(str(data_location / 'gridMET' / f'*{var}*.nc'))
    obs = xr.open_dataset(obs_path[0], decode_times = True)

    raw_path = glob.glob(str(data_location / 'daily' / f'*{var}*.nc'))
    raw = xr.open_dataset(raw_path[0], decode_times = True)

    # Pass datasets to debiaser
    test = ECDFM(obs, raw, var)

    # Create output directory to store new cleaned files
    output_dir = current_dir / 'wrfout' 
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Generate save name for debiased data
    out_path = os.path.join(output_dir, f'wrfout_GSLBIP_multimodel_ssp245_{var}.nc')

    # Save data to netCDF file
    test.to_netcdf(out_path)
    logger.success(f'File saved to: {out_path}')

    # Close out of data once saved
    test.close()

if __name__ == '__main__':

    # Track program time in log files
    start = time.perf_counter()
    logger.info('Beginning execution.')

    # Only inputs required
    main(
            var = 'tmmx',
            data_location = current_dir
            )

    # Report of runtime at completion 
    logger.success(f'Debiasing process completed!')
    logger.info(f'Total runtime: {time.perf_counter() - start:.4f}s')

    # Force script to stop running once code is finished
    sys.exit(0)
    


