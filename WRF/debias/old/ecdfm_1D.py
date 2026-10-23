"""
Author: Sydney Smith
Date Created: October 23, 2026
"""

from datetime import datetime
import glob
from loguru import logger
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
from statsmodels.distributions.empirical_distribution import StepFunction
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

def invert_ecdf(ecdf_obj):
    """
    Inverts a statsmodels ECDF object using its .y and .x attributes, regardless of if
    the values are monotonic or not, with safe extrapolation to prevent NaNs.
    """
    # ECDF from statsmodels sets first x value to -inf which causes NaNs for very small percentiles passed to the inverted function
    # Mask out any indefinite values to prevent possible NaNs
    finite_mask = np.isfinite(ecdf_obj.x)
    x_vals = ecdf_obj.x[finite_mask]
    y_vals = ecdf_obj.y[finite_mask]

    # Ensure x_vals are unique and strictly increasing for interp1d (monotonic)
    x_unique, indices = np.unique(x_vals, return_index=True)
    y_unique = y_vals[indices]

    return interp1d(
        y_unique, # Pass y values as x values
        x_unique, # And vice versa
        bounds_error = False, 
        fill_value = 'extrapolate'  # Prevents NaNs if raw percentiles push past bounds
    )

def apply_to_date(obs: xr.Dataset, hist, mod, var, date) -> np.ndarray:
    """
    Debiases data for a given date by constructing a CDF using a dataset restricted to be within a 
    running window. Date must be a valid date that exists within the dataset formatted as '%Y-%m-%d' 
    datetime object.
    """
    # TODO: check how running_window_slice handles leap years (try to set calendar year to 1988 instead since it's a leap year)

    # Slice data into 31 day running windows centered on given date (note: 31 days is defualt setting)
    obs_sliced = running_window_slice(obs, date)
    logger.info(f'NaNs in obs running window: {float(np.isnan(obs_sliced).sum())}')

    hist_sliced = running_window_slice(hist, date)
    logger.info(f'NaNs in hist running window: {float(np.isnan(hist_sliced).sum())}')

    mod_sliced = running_window_slice(mod, date)
    logger.info(f'NaNs in mod running window: {float(np.isnan(mod_sliced).sum())}')
    logger.info('Datasets sliced into running windows.')

    # End function if NaNs are present in the source data
    if np.isnan(obs_sliced).any() or np.isnan(hist_sliced).any() or np.isnan(mod_sliced).any():
        logger.error('NaNs detected in source data.')
        return NotImplementedError

    # Create CDFs from sliced datasets
    obs_cdf = ECDF(obs_sliced)
    hist_cdf = ECDF(hist_sliced)
    mod_cdf = ECDF(mod_sliced)

    # Select day of interest and create mask
    day = int(date.strftime('%d'))
    month = int(date.strftime('%m'))
    mask = (mod.time.dt.month == month) & (mod.time.dt.day == day)

    # Skip dates that don't exist
    if not mask.any():
        logger.warning(f'No date found for {date}.')
        return None
    
    # Apply mask to dataset
    mod_date = mod.where(mask, drop = True) # This should select day of interest across all years in ds
    logger.info(f'Dataset sliced down to date of interest: {date}')
    logger.info(mod_date)

    # Use mod_cdf like a function to find all of the percentiles in mod_date
    mod_percentiles = mod_cdf(mod_date)
    logger.info(f'Percentile values for mod data: {mod_percentiles}')
    logger.info(f'NaNs in mod_percentiles: {float(np.isnan(mod_percentiles).sum())}')

    # TODO: NaNs are appearing in obs and hist from this point on
    # Invert hist_cdf and obs_cdf to take a percentile and output a data point
    inv_obs = invert_ecdf(obs_cdf)
    inv_hist = invert_ecdf(hist_cdf)

    # Find data points for every percentile in mod_percentiles
    obs_vals = inv_obs(mod_percentiles)
    logger.info(f'NaNs in obs_vals: {float(np.isnan(obs_vals).sum())}')
    logger.info(f'Obs values for given percentiles: {obs_vals}')

    hist_vals = inv_hist(mod_percentiles)
    logger.info(f'NaNs in hist_vals: {float(np.isnan(hist_vals).sum())}')
    logger.info(f'Historical values for given percentiles: {hist_vals}')

    # Perform bias adjustment
    bias = hist_vals - obs_vals
    logger.info(f'NaNs in bias: {float(np.isnan(bias).sum())}')
    mod_bias_corrected = mod_date[var].values - bias

    # End script early if there are Nans in data
    if np.isnan(mod_bias_corrected).any():
        logger.error(f'NaNs in debiased data: {float(np.isnan(mod_bias_corrected).sum())}')
        logger.info(f'Nan error occured on {date}.')
        return None

    logger.info('Mod data successfully bias corrected!')
    logger.info(mod_bias_corrected)
    logger.info(f'Bias corrected max: {mod_bias_corrected.max()}')
    logger.info(f'Bias corrected min: {mod_bias_corrected.min()}')

    # Close out of unwanted datasets
    obs_sliced.close()
    hist_sliced.close()
    mod_sliced.close()
    mod_date.close()
    del obs_vals
    del hist_vals
    del mod_percentiles
    gc.collect()

    return mod_bias_corrected

def ECDFM(obs, hist, mod, var, lat = 40.788, lon = -111.978):
    """
    Performs ECDFM quantile mapping debiasing process on given data based on data from a given histroical
    period and corresponding observational data. Model data should include both historical and future data. 
    """

    # Select specific location of interest (choses nearest possible location)
    obs = loc_sel(obs, var, lat = lat, lon = lon)
    logger.info(f'NaNs at selected obs location: {float(obs[var].isnull().sum().values)}')
    hist = loc_sel(hist, var, lat = lat, lon = lon)
    logger.info(f'NaNs at selected hist location: {float(hist[var].isnull().sum().values)}')
    mod = loc_sel(mod, var, lat = lat, lon = lon)
    logger.info(f'NaNs at selected mod location: {float(mod[var].isnull().sum().values)}')
    logger.info(mod)

    # Make copy of mod_loc to plug debiased data into
    template = mod.copy(deep = True)

    # Select first and last day in dataset
    first_date = pd.to_datetime(mod.isel(time = 0).time.values)
    last_date = pd.to_datetime(mod.isel(time = -1).time.values)

    # Create date range that spans a calendar year (select a year that is a leap year)
    dates = pd.date_range(start = f"1988-{first_date.strftime('%m-%d')}", end = f"1988-{last_date.strftime('%m-%d')}", freq = 'D') 

    for date in dates:
        # Apply bias correction for single date
        logger.info(f'Date type: {type(date)}')

        # Debias the given day of year
        debiased_doy = apply_to_date(obs, hist, mod, var, date)

        # Exit out of function if Nans occur
        if debiased_doy is None:
            logger.error(f'Error detected in dataset. See log above for more information.')
            return None

        # Pull month and day from date
        day = int(date.strftime('%d'))
        month = int(date.strftime('%m'))

        # Plug in debiased data
        mask = (template.time.dt.month == month) & (template.time.dt.day == day)
        template[var].data[mask] = debiased_doy 
        logger.info(f'Debiased data for {date} added to dataset!')

        # Back into data's bias to ensure that the bias corrected data doesn't equal to mod data
        temp_masked = template[var].where(mask, drop = True)
        mod_masked = mod[var].where(mask, drop = True)
        bias = mod_masked - temp_masked # Back into data's bias
        logger.info(f'bias type: {type(bias)}')

        # Create boolean to check if bias is all zeros
        if isinstance(bias, xr.Dataset):
            is_all_zeros = (bias[var] == 0).all().item() # Returns True if every single value is zero
            bias = bias[var]
        elif isinstance(bias, xr.DataArray): 
            # Expects bias to be xr.DataArray
            is_all_zeros = (bias == 0).all().item() # Returns True if every single value is zero
        else: 
            logger.error(f'Unexpected type for bias: {type(bias)}')
            return None

        # Ensure that bias is not all zeros
        if is_all_zeros: # Triggers if bias_check is set to true
            logger.info(bias)
            logger.error('Bias corrected data was not successfully applied to dataset.')
            return None
        else:
            logger.info(f'Bias for {date} successfully corrected and saved to dataset.')

        # Close unnecessay datasets
        temp_masked.close()
        mod_masked.close()

    return template

def detrend_ds(ds, var, deg = 1):
    """
    Detrended a given 3D dataset along the time dimension.
    """
    # Calculate the data's mean
    ds_avg = ds[var].mean(dim = 'time')

    # End function if NaNs are present in the source data
    if ds['time'].isnull().any():
        logger.error('NaNs detected in source data.')
        return NotImplementedError

    # Detrend the mod model data
    ds_fit_line = ds.polyfit(dim = 'time', deg = deg) # One degree is for a linear fit
    
    # Create the trend line based on fitted slope and y-intercept at each location
    ds_trend = xr.polyval(ds['time'], ds_fit_line[f'{var}_polyfit_coefficients'])

    # Subtract the trend from the dataset and add back its mean
    detrended_ds = ds - ds_trend
    detrended_with_mean = detrended_ds + ds_avg

    # Close out of unnecessary datasets
    ds_fit_line.close()
    ds.close()

    return detrended_with_mean, ds_trend, ds_avg

def retrend_ds(ds, var, trend, ds_mean):
    """
    Retrend data based on the detrending process from the function above.
    """
    # Start by removing the mean that was added to the dataset
    mean_out = ds - ds_mean

    # Retrended the data using its fit line
    retrended_ds = mean_out + trend

    return retrended_ds

# Catch silent errors and report to log file
@logger.catch 
def main(var, data_location, save = True):
    # Open datasets
    obs_path = glob.glob(str(data_location / 'gridMET' / f'*{var}*.nc'))
    obs = xr.open_dataset(obs_path[0], decode_times = True)

    mod_path = glob.glob(str(data_location / 'daily' / f'*{var}*.nc'))
    mod = xr.open_dataset(mod_path[0], decode_times = True)

    # Slice model data down to historical and future period respectively
    hist = mod.sel(time = slice('1985-01-01', '2014-12-31'))
    fut = mod.sel(time = slice('2015-01-01', '2099-12-31'))

    # Set location of interest as SLC airport
    lat = 40.788
    lon = -111.978
    
    # Pass datasets to debiaser 
    debiased_fut = ECDFM(obs, hist, fut, var, lat, lon)

    # Intermediate save for future data
    if save:
        # Create output directory to store new cleaned files
        output_dir = current_dir / 'wrfout_sep'
        os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

        # Generate save name for debiased data
        debiased_fut_path = os.path.join(output_dir, f'debiased_fut_{var}.nc')
  
        # Save data to netCDF file
        debiased_fut.to_netcdf(debiased_fut_path)
        logger.success(f'Future files saved to: {output_dir}')

    # Close out of unnecessary data
    debiased_fut.close()

    # Pass datasets to debiaser 
    debiased_hist = ECDFM(obs, hist, hist, var, lat, lon)

    if save:
        # Create output directory to store new cleaned files
        output_dir = current_dir / 'wrfout_sep'
        os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

        # Generate save name for debiased data
        debiased_hist_path = os.path.join(output_dir, f'debiased_hist_{var}.nc')

        # Save data to netCDF file
        debiased_hist.to_netcdf(debiased_hist_path)
        logger.success(f'Historical files saved to: {output_dir}')

    # Close out of data
    debiased_hist.close()

if __name__ == '__main__':

    # Track program time in log files
    start = time.perf_counter()
    logger.info('Beginning execution.')

    # Only inputs required
    main(
            var = 'tmmx',
            data_location = current_dir,
            save = True
            )

    # Report of runtime at completion 
    logger.success(f'Debiasing process completed!')
    logger.info(f'Total runtime: {time.perf_counter() - start:.4f}s')

    # Force script to stop running once code is finished
    sys.exit(0)

