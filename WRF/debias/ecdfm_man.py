"""
Author: Sydney Smith
Date Created: September 29, 2026
"""

from datetime import datetime
import gc
import glob
from loguru import logger
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
from statsmodels.distributions.empirical_distribution import (
    StepFunction)
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

def apply_to_date(obs: xr.Dataset, hist, raw, var, date) -> np.ndarray:
    """
    Debiases data for a given date by constructing a CDF using a dataset restricted to be within a 
    running window. Date must be a valid date that exists within the dataset formatted as '%Y-%m-%d' 
    datetime object.
    """
    # TODO: check how running_window_slice handles leap years (try to set calendar year to 1988 instead since it's a leap year)

    # Slice data into 31 day running windows centered on given date (note: 31 days is defualt setting)
    obs_sliced = running_window_slice(obs, date)
    logger.info(f'NaNs in obs running window: {float(obs_sliced[var].isnull().sum().values)}')

    hist_sliced = running_window_slice(hist, date)
    logger.info(f'NaNs in hist running window: {float(hist_sliced[var].isnull().sum().values)}')

    raw_sliced = running_window_slice(raw, date)
    logger.info(f'NaNs in raw running window: {float(raw_sliced[var].isnull().sum().values)}')
    logger.info('Datasets sliced into running windows.')

    # End function if NaNs are present in the source data
    if obs_sliced[var].isnull().any() or hist_sliced[var].isnull().any() or raw_sliced[var].isnull().any():
        logger.error('NaNs detected in source data.')
        return NotImplementedError

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
    logger.info(f'NaNs in raw_percentiles: {float(np.isnan(raw_percentiles).sum())}')

    # TODO: NaNs are appearing in obs and hist from this point on
    # Invert hist_cdf and obs_cdf to take a percentile and output a data point
    inv_obs = invert_ecdf(obs_cdf)
    inv_hist = invert_ecdf(hist_cdf)

    # Find data points for every percentile in raw_percentiles
    obs_vals = inv_obs(raw_percentiles)
    logger.info(f'NaNs in obs_vals: {float(np.isnan(obs_vals).sum())}')
    logger.info(f'Obs values for given percentiles: {obs_vals}')

    hist_vals = inv_hist(raw_percentiles)
    logger.info(f'NaNs in hist_vals: {float(np.isnan(hist_vals).sum())}')
    logger.info(f'Historical values for given percentiles: {hist_vals}')

    # Perform bias adjustment
    bias = hist_vals - obs_vals
    logger.info(f'NaNs in bias: {float(np.isnan(bias).sum())}')
    raw_bias_corrected = raw_date[var].values - bias

    # End script early if there are Nans in data
    if np.isnan(raw_bias_corrected).any():
        logger.error(f'NaNs in debiased data: {float(np.isnan(raw_bias_corrected).sum())}')
        logger.info(f'Nan error occured on {date}.')
        return None

    logger.info('Raw data successfully bias corrected!')
    logger.info(raw_bias_corrected)
    logger.info(f'Bias corrected max: {raw_bias_corrected.max()}')
    logger.info(f'Bias corrected min: {raw_bias_corrected.min()}')

    # Close out of unwanted datasets
    obs_sliced.close()
    hist_sliced.close()
    raw_sliced.close()
    raw_date.close()
    del obs_vals
    del hist_vals
    del raw_percentiles
    gc.collect()

    return raw_bias_corrected

def ECDFM(obs, raw, var, lat, lon):
    """
    Performs ECDFM quantile mapping debiasing process on given data based on data from a given histroical
    period and corresponding observational data. Raw data should include both historical and future data. 
    """

    # Slice raw data down to historical period
    hist = raw.sel(time = slice('1985-01-01', '2014-12-31'))
    
    # Select specific location of interest (choses nearest possible location)
    obs_loc = loc_sel(obs, lat = lat, lon = lon)
    logger.info(f'NaNs at selected obs location: {float(obs_loc[var].isnull().sum().values)}')
    hist_loc = loc_sel(hist, lat = lat, lon = lon)
    logger.info(f'NaNs at selected hist location: {float(hist_loc[var].isnull().sum().values)}')
    raw_loc = loc_sel(raw, lat = lat, lon = lon)
    logger.info(f'NaNs at selected raw location: {float(raw_loc[var].isnull().sum().values)}')
    logger.info(raw_loc[var])

    # Make copy of raw_loc to plug debiased data into
    template = raw_loc.copy()

    # Create date range that spans a calendar year (select a year that is a leap year)
    dates = pd.date_range(start = '1988-01-01', end = '1988-12-31', freq = 'D') 

    for date in dates:
        # Apply bias correction for single date
        logger.info(f'Date type: {type(date)}')
        debiased_doy = apply_to_date(obs_loc, hist_loc, raw_loc, var, date)

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

        # Back into data's bias to ensure that the bias corrected data doesn't equal to raw data
        temp_masked = template[var].where(mask, drop = True)
        raw_masked = raw[var].where(mask, drop = True)
        bias = raw_masked - temp_masked # Back into data's bias
        logger.info(f'bias type: {type(bias)}')

        # Ensure that bias is not all zeros
        if (bias == 0).all(): # Ensure it's not all zeros
            logger.error('Bias corrected data was not successfully applied to dataset.')
            return None
        else:
            logger.info(f'Bias for {date} successfully corrected.')

        # Close unnecessay datasets
        temp_masked.close()
        raw_masked.close()

    return template

def nan_test(var):
    """
    Returns location of Nans in debiased data.
    """
    
    fpath = glob.glob(str(current_dir / 'wrfout' / f'*{var}*.nc'))
    debiased = xr.open_dataset(fpath[0], decode_times = True)

    nan_data = debiased[var].where(debiased[var].isnull(), drop = True)

    return nan_data

def np_nan_loc(arr):
    """
    Find locations of NaNs in np.array.
    """
    return np.argwhere(np.isnan(arr))

# Catch silent errors and report to log file
@logger.catch 
def main(var, data_location, save = True):
    # Open datasets
    obs_path = glob.glob(str(data_location / 'gridMET' / f'*{var}*.nc'))
    obs = xr.open_dataset(obs_path[0], decode_times = True)

    raw_path = glob.glob(str(data_location / 'daily' / f'*{var}*.nc'))
    raw = xr.open_dataset(raw_path[0], decode_times = True)

    # Detrend the raw model data
    fit_line = raw.polyfit(dim = 'time', deg = 1) # deg = 1 is for a linear fit line
    
    # Create the trend line based on fitted slope and y-intercept at each location
    trend = xr.polyval(raw['time'], fit_line['tmmx_polyfit_coefficients'])

    # Subtract the trend from the dataset
    detrended = raw - trend

    # Set location of interest as SLC airport
    lat = 40.788
    lon = -111.978

    # Pass datasets to debiaser
    debiased = ECDFM(obs, detrended, var, lat, lon)

    # TODO: retrend data
    # retrended = debiased + trend

    if save:
        # Create output directory to store new cleaned files
        output_dir = current_dir / 'wrfout' 
        os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

        # Generate save name for debiased data
        out_path = os.path.join(output_dir, f'wrfout_GSLBIP_multimodel_ssp245_{var}_test.nc')

        # Save data to netCDF file
        retrended.to_netcdf(out_path)
        logger.success(f'File saved to: {out_path}')

    # Close out of data
    debiased.close()
    # retrended.close()

# if __name__ == '__main__':

#     # Track program time in log files
#     start = time.perf_counter()
#     logger.info('Beginning execution.')

#     # Only inputs required
#     main(
#             var = 'tmmx',
#             data_location = current_dir,
#             save = True
#             )

#     # Report of runtime at completion 
#     logger.success(f'Debiasing process completed!')
#     logger.info(f'Total runtime: {time.perf_counter() - start:.4f}s')

#     # Force script to stop running once code is finished
#     sys.exit(0)

# TODO: create vectorized approach to debiasing instead of slicing to a specific location


var = 'tmmx'
date = pd.to_datetime('2001-03-05')
lat = 40.788
lon = -111.978

raw_path = glob.glob(str(current_dir / 'daily' / f'*{var}*.nc'))
raw = xr.open_dataset(raw_path[0], decode_times = True)

# Slice data into 31 day running windows centered on given date (note: 31 days is defualt setting)
# obs_sliced = running_window_slice(obs, date)
# logger.info(f'NaNs in obs running window: {float(obs_sliced[var].isnull().sum().values)}')

# hist_sliced = running_window_slice(hist, date)
# logger.info(f'NaNs in hist running window: {float(hist_sliced[var].isnull().sum().values)}')

raw_sliced = running_window_slice(raw, date)
logger.info(f'NaNs in raw running window: {float(raw_sliced[var].isnull().sum().values)}')
logger.info('Datasets sliced into running windows.')

# # End function if NaNs are present in the source data
# if obs_sliced[var].isnull().any() or hist_sliced[var].isnull().any() or raw_sliced[var].isnull().any():
#     logger.error('NaNs detected in source data.')

# Create CDFs from sliced datasets
# obs_cdf = ECDF(obs_sliced[var].values.flatten())
# hist_cdf = ECDF(hist_sliced[var].values.flatten())
# raw_cdf = ECDF(raw_sliced[var].values.flatten())

class StepFunc3D:
    def __init__(self, x, y, ival = 0, sorted = False, side = 'left'):
        if side.lower() not in ['right', 'left']:
            raise ValueError('Side can take the values "right" or "left."')
        self.side = side

        self.x = np.asarray(x, dtype = float)
        self.y = np.asarray(y, dtype = float)
        self.ival = ival

        if self.x.shape != self.y.shape:
            raise ValueError('x and y must have the same shape.')
        if self.x.ndim < 1:
            raise ValueError('x and y must have at least 1 dimension.')

        if not sorted: 
            self.x = np.sort(self.x, axis = 0)
            self.y = np.sort(self.y, axis = 0)

        self.n_steps = self.x.shape[0]

    def __call__(self, time):
        time = np.asarray(time, dtype=float)
        if time.ndim == self.x.ndim - 1:             
            time = time[None]
        if time.shape[1:] != self.x.shape[1:]:
            raise ValueError(f'Spatial dims must be {self.x.shape[1:]}, got {time.shape[1:]}.')

        below = np.less if self.side == 'left' else np.less_equal
        n = self.n_steps

        count = np.zeros(time.shape, dtype=np.intp)
        for step in 2 ** np.arange(int(np.log2(n)), -1, -1):
            probe = count + step
            x_probe = np.take_along_axis(self.x, np.minimum(probe, n) - 1, axis=0)
            count = np.where((probe <= n) & below(x_probe, time), probe, count)

        tind = count - 1                                
        result = np.take_along_axis(self.y, np.maximum(tind, 0), axis=0)
        result = np.where(tind < 0, self.ival, result)
        return np.where(np.isnan(time), np.nan, result)

class vectorized_ECDF(StepFunc3D):
    """
    X Dimension: sorted data values
    Y Dimension: cumulative probabilities ranging from 1/N to 1
    """
    def __init__(self, data, axis = 0, side = 'right'):
        arr = np.asarray(data, dtype = float)

        if axis != 0:
            arr = np.moveaxis(arr, source = axis, destination = 0)

        nobs = arr.shape[0]
        sorted_x = np.sort(arr, axis = 0)

        percentile_1d = np.arange(1, nobs +1, dtype = float) / nobs
        broadcast_shape = (nobs,) + (1,) * (sorted_x.ndim -1)
        sorted_y = np.broadcast_to(percentile_1d.reshape(broadcast_shape), sorted_x.shape).copy()

        super().__init__(x = sorted_x, y = sorted_y, ival = 0, sorted = True, side = side)



ecdf_func = vectorized_ECDF(raw_sliced[var].values, axis=0)

day = int(date.strftime('%d'))
month = int(date.strftime('%m'))
mask = (raw.time.dt.month == month) & (raw.time.dt.day == day)

# Skip dates that don't exist
if not mask.any():
    logger.warning(f'No date found for {date}.')

# Apply mask to dataset
raw_date = raw.where(mask, drop = True)

probability_grid = ecdf_func(raw_date[var].values)


# Use raw_cdf like a function to find all of the percentiles in raw_date
raw_percentiles = raw_cdf(raw_date[var].values)
logger.info(f'Percentile values for raw data: {raw_percentiles}')
logger.info(f'NaNs in raw_percentiles: {float(np.isnan(raw_percentiles).sum())}')

# TODO: NaNs are appearing in obs and hist from this point on
# Invert hist_cdf and obs_cdf to take a percentile and output a data point
inv_obs = invert_ecdf(obs_cdf)
inv_hist = invert_ecdf(hist_cdf)

# Find data points for every percentile in raw_percentiles
obs_vals = inv_obs(raw_percentiles)
logger.info(f'NaNs in obs_vals: {float(np.isnan(obs_vals).sum())}')
logger.info(f'Obs values for given percentiles: {obs_vals}')

hist_vals = inv_hist(raw_percentiles)
logger.info(f'NaNs in hist_vals: {float(np.isnan(hist_vals).sum())}')
logger.info(f'Historical values for given percentiles: {hist_vals}')

# Perform bias adjustment
bias = hist_vals - obs_vals
logger.info(f'NaNs in bias: {float(np.isnan(bias).sum())}')
raw_bias_corrected = raw_date[var].values - bias

# End script early if there are Nans in data
if np.isnan(raw_bias_corrected).any():
    logger.error(f'NaNs in debiased data: {float(np.isnan(raw_bias_corrected).sum())}')
    logger.info(f'Nan error occured on {date}.')
    return None

logger.info('Raw data successfully bias corrected!')
logger.info(raw_bias_corrected)
logger.info(f'Bias corrected max: {raw_bias_corrected.max()}')
logger.info(f'Bias corrected min: {raw_bias_corrected.min()}')

# Close out of unwanted datasets
obs_sliced.close()
hist_sliced.close()
raw_sliced.close()
raw_date.close()
del obs_vals
del hist_vals
del raw_percentiles
gc.collect()
