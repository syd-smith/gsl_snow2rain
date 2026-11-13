#%%
"""
Author: Sydney Smith
Date Created: August 25, 2026
"""

from datetime import datetime
import glob
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
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
parent_dir = current_dir.parent

# Import custom python modules from other directories
sys.path.append(str(parent_dir.parent.parent))
from from_savanna.nclcmaps import cmap

sys.path.append(str(parent_dir))
from old.temporal_chunks import open_or_skip, get_fpaths

sys.path.append(str(current_dir))
from preprocessing import fix_time_coord
from qdm_man import running_window_slice, loc_sel

# ===================
# - Set Up Logger - 
# ===================

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

units = {
    'tmmn': 'K',
    'tmmx': 'K',
    'pr': 'mm/day',
    'sph': 'kg kg-1',
    'srad': 'W m-2',
    'vas': 'm s-1',
    'uas': 'm s-1',
}

title = {
    'tmmn': 'Minimum Temperature',
    'tmmx': 'Maximum Temperatute',
    'pr': 'Precipitation',
    'sph': 'Specific Humidity',
    'srad': 'Short Wave Radiation',
    'vas': 'Northward Wind',
    'uas': 'Eastward Wind',
}

# =====================
# ---- Functions ----
# =====================

def best_fit(var, data, deg = 1):
    """
    Calculate the line of best fit for plotting. 
    """
    # Check that input data is in correct format
    if isinstance(data, xr.Dataset):
        data = data[var]

    elif not isinstance(data, xr.DataArray):
        raise ValueError('Input data must be either an xr.Dataset or xr.DataArray.')

    # Turn timestamps into numerical values
    x_vals = list(range(0, data.time.shape[0]))

    # Calculate the slope and intercept for the data
    ds_slope, ds_intercept = np.polyfit(x_vals, data.values, deg = deg)
    ds_slope = np.full(len(x_vals), ds_slope)
    ds_intercept = np.full(len(x_vals), ds_intercept)

    # Make y values for the line of best fit
    best_line_y_vals = ds_slope * x_vals + ds_intercept 

    return best_line_y_vals


def trend_plt(var, obs, hist, fut, db_hist, db_fut, date, lat, lon, save_name = 'tend_plot', save = False):
    """
    Create a graph of model WRF output data, observational data, and both detrended datasets to verify detrending 
    was successful. Data plotted is from a single location and a 31 day running window centered around a given date.
    """
    logger.info(f'Creating detrending verification plot for {var}.')

    # Initialize plot
    fig, ax = plt.subplots(1, 1, figsize = (12, 6), sharex = True)

    # Slice data down to a location of interest
    obs_loc = loc_sel(obs, var, lat, lon)
    hist_loc = loc_sel(hist, var, lat, lon)
    fut_loc = loc_sel(fut, var, lat, lon)
    db_hist_loc = loc_sel(db_hist, var, lat, lon)
    db_fut_loc = loc_sel(db_fut, var, lat, lon)

    # Resample the data to yearly averages for trend calculations
    obs_yearly = obs_loc.resample(time = '1YS').mean()
    hist_yearly = hist_loc.resample(time = '1YS').mean()
    fut_yearly = fut_loc.resample(time = '1YS').mean()
    db_hist_yearly = db_hist_loc.resample(time = '1YS').mean()
    db_fut_yearly = db_fut_loc.resample(time = '1YS').mean()
  
    logger.info(f'Location slicing and date selection completed for {var} detrend plot.')
    
    # Graph obs data
    ax.plot(
        obs_yearly.time,
        obs_yearly.values, 
        'k-', 
        alpha = 0.6, 
        label = 'Obs'
    )

    ax.plot(
            obs_yearly.time,
            best_fit(var, obs_yearly), 
            'k-', 
            alpha = 0.9, 
        )

    # Graph raw WRF output data
    ax.plot(
        hist_yearly.time, 
        hist_yearly.values, 
        'r-', 
        alpha = 0.3,
        label = 'WRF'
    )

    ax.plot(
        hist_yearly.time,
        best_fit(var, hist_yearly), 
        'r-', 
        alpha = 0.9, 
    )

    ax.plot(
        fut_yearly.time, 
        fut_yearly.values, 
        'r-', 
        alpha = 0.3,
        label = 'WRF'
    )

    ax.plot(
        fut_yearly.time,
        best_fit(var, fut_yearly), 
        'r-', 
        alpha = 0.9, 
    )

    # Graph debiased data
    ax.plot(
        db_hist_yearly.time,
        db_hist_yearly.values, 
        'g-',
        alpha = 0.3, 
        label = 'Detrended Obs'
    )

    ax.plot(
        db_hist_yearly.time,
        best_fit(var, db_hist_yearly), 
        'g-', 
        alpha = 0.9, 
    )

    ax.plot(
            db_fut_yearly.time,
            db_fut_yearly.values, 
            'g-',
            alpha = 0.3, 
            label = 'Detrended Obs'
        )
    
    ax.plot(
        db_fut_yearly.time,
        best_fit(var, db_fut_yearly),
        'g-', 
        alpha = 0.9, 
    )

    # Adjust plot format settings
    # ax.set_title(f'Trended vs. Detrended {title[var]} Data')
    ax.set_ylabel(f'{var} ({units[var]})')
    ax.set_xlabel('Time')
    ax.grid(True, linestyle = '--', alpha = 0.5)
    ax.legend(frameon = True)

    # Set labels for the x axis
    ticks = [f'{yr}-01-01T00:00:00.000000000' for yr in range(1985, 2100, 5)]
    labels = [f'{yr}' for yr in range(1985, 2100, 5)]
    ax.set_xticks(ticks = ticks, labels = labels)
    ax.tick_params(axis = 'x', rotation = 45)

    plt.tight_layout()
    plt.show()
    logger.info(f'Trend plot complete for {var}.')

    # Opt to save image to sub directory
    if save:
        os.makedirs(current_dir / 'figures' / var, exist_ok = True)
        save_path = current_dir / 'figures' / var / f'{save_name}.png'
        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Trend plot saved to: {save_path}')

def sample(data, var):
    """ 
    Take a random sample from the given data that is 5% of its original size. 
    """

    # Flatten the data into a 1D array
    to_np = data[var].to_numpy()
    flattened = to_np.flatten()

    # Set size of sample dataset relative to area size
    sample_size = int(len(flattened) * 0.05)

    # Produce sample array
    rng = np.random.default_rng()
    sample = rng.choice(flattened, size = sample_size)
    logger.info(f'Random sample of {var} taken from dataset. Sample size: {sample_size}.')

    return sample

def cdf_masked(data):
    """
    Mask out long tails in cdf data.
    """
    logger.info(f'Data.x before masking: {data.x}')
    logger.info(f'Data.y before masking: {data.y}')

    # Generate mask with bounds at 1% and 99%
    mask = (data.y >= 0.01) & (data.y <= 0.99)

    # Apply to data if mask is generated
    if mask.any():
        masked_x = data.x[mask]
        masked_y = data.y[mask]

        logger.info(f'Data.x after masking: {masked_x}')
        logger.info(f'Data.y after masking: {masked_y}')
        return masked_x, masked_y

    # Otherwise log error
    else:
        logger.error('Failed to generate mask for data.')
        return

def cdf_plt(var, obs, mod, db_hist, db_fut, dates, lat, lon, save_name, save = True):
    """
    Create a plot of the cumulative distribution function for each of the given datasets.
    Data will be reduced to the given locations and dates. Dates must be str(%Y/%m/%d).
    This function was intended to create a four pannel plot relecting four dates passed 
    with one date in each season (a seasonal pannel cdf plot).
    """
    import math
    from statsmodels.distributions.empirical_distribution import ECDF
    logger.info(f'Creating cdf plot for {var}.')

    # Adjust plot size to number of dates passed
    if len(dates) > 3:
        # Make additional rows if stats list is too long
        fig, ax = plt.subplots(int(len(dates)/2), math.ceil(len(dates)/2), figsize = (10, 10)) # Round up the number of columns needed to the nearest whole number
        ax = ax.flatten() # Flatten the axes array to make it easier to iterate over
    
    # Keep figure to one row if dates list is 3 items or less
    else:
        fig, ax = plt.subplots(1, len(dates), figsize = (10, 10))

    # Slice dataset to a specific location
    obs_sel = loc_sel(obs, var, lat, lon)
    mod_sel = loc_sel(mod, var, lat, lon)
    db_hist_sel = loc_sel(db_hist, var, lat, lon)
    db_fut_sel = loc_sel(db_fut, var, lat, lon)

    # List of plot labels
    labels = ['a.', 'b.', 'c.', 'd.']

    for (i, date), label in zip(enumerate(dates), labels):
        # Convert date to pd.to_datetime
        date = pd.to_datetime(date)

        # Slice dataset down to a given window size centered on date
        # Default window size is 31 days
        window_size = 31
        obs_window = running_window_slice(obs_sel, date, window_size)
        mod_window = running_window_slice(mod_sel, date, window_size)
        db_hist_window = running_window_slice(db_hist_sel, date, window_size)
        db_fut_window = running_window_slice(db_fut_sel.copy(deep = True), date, window_size)

        # Use to check that means from the window slices are different for each date used
        logger.info(f"Date: {date.strftime('%Y-%m-%d')} | Obs Mean Temp: {obs_window.mean().values}")
        logger.info(f"Date: {date.strftime('%Y-%m-%d')} | Mod Mean Temp: {mod_window.mean().values}")
        logger.info(f"Date: {date.strftime('%Y-%m-%d')} | DB Hist Mean Temp: {db_hist_window.mean().values}")
        logger.info(f"Date: {date.strftime('%Y-%m-%d')} | DB fut Mean Temp: {db_fut_window.mean().values}")

        # Pull the year from date
        year = int(date.strftime('%Y'))

        # Slice into historical and future periods
        hist_window = mod_window.sel(time = mod_window.time.dt.year.isin(range(1985, 2015)))
        fut_window = mod_window.sel(time = mod_window.time.dt.year.isin(range(2015, 2099)))

        # Calculate a CDF for each array
        obs_cdf = ECDF(obs_window)
        hist_cdf = ECDF(hist_window)
        fut_cdf = ECDF(fut_window)
        db_hist_cdf = ECDF(db_hist_window)
        db_fut_cdf = ECDF(db_fut_window)
        logger.info(f'CDFs calculated for {date}.')

        # Mask CDF outputs
        obs_x, obs_y = cdf_masked(obs_cdf)
        raw_hist_x, raw_hist_y = cdf_masked(hist_cdf)
        raw_fut_x, raw_fut_y = cdf_masked(fut_cdf)
        debiased_hist_x, debiased_hist_y = cdf_masked(db_hist_cdf)
        debiased_fut_x, debiased_fut_y = cdf_masked(db_fut_cdf)

        # Plot CDFs
        ax[i].plot(
            obs_x, 
            obs_y, 
            'k-',
            label = 'Observations', 
            rasterized = True
        )

        ax[i].plot(
            raw_hist_x,
            raw_hist_y,
            'r-.',
            label = 'Historical WRF Output', 
            rasterized = True
        )

        ax[i].plot(
            raw_fut_x, 
            raw_fut_y, 
            'r-',
            label = 'Future WRF Output',
            rasterized = True
        )

        ax[i].plot(
            debiased_hist_x,
            debiased_hist_y,
            'g-.',
            label = 'Historical Debiased WRF',
            rasterized = True
        )

        ax[i].plot(
            debiased_fut_x, 
            debiased_fut_y, 
            'g-',
            label = 'Future Debiased WRF',
            rasterized = True
        )

        # Adjust plot format settings for each subplot
        ax[i].text(15, 0.02, label, fontsize = 20)
        ax[i].set_ylabel('Percentile')
        ax[i].set_xlabel(f'{var} ({units[var]})')
        ax[i].grid(True, linestyle = '--', alpha = 0.5)
        logger.info(f'Plotting completed for {date}.')

    # Global plot settings
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper left', bbox_to_anchor = (0.95, 0.95), frameon = True)
    # plt.tight_layout()
    plt.show()
    logger.info(f'CDF plot complete for {var}.')

    # Opt to save image to sub directory
    if save:
        os.makedirs(current_dir / 'figures' / var, exist_ok = True)
        save_path = current_dir / 'figures' / var / save_name
        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'CDF plot saved to: {save_path}')

def calc_stat(data, var, stat = 'mean'):
    """"
    Average data based on the specified statistic. Accepts xr.DataArray as input.
    """
    logger.info(type(data))

    # Group the data by day of year and calculate the specified statistic (mean, median, 0.05, or 0.95) for each day of the year
    try:
        q_val = float(stat)
        logger.info(f'Quantile value of {q_val} selected for {var}.')

        # Apply quantile function
        dayOyear = data.quantile(q_val)
        logger.success(f'Quantile of {q_val} successfully calculated!')

    # Check for mean or median if stat specificed is not a float
    except (ValueError, TypeError):
        if stat == 'mean':
            dayOyear = data.mean()
            logger.success('Mean of data successfully calculated!')

        elif stat == 'median':
            dayOyear = data.median()
            logger.success('Median of data successfully calculated!')

        else:
            logger.error(f'{stat} is not a valid statistic. Please choose from mean, median, or a float between 0 and 1.')

    logger.info(f'calc_stat returns {float(dayOyear.values)}.')
    logger.info(f'Day of year {stat} complete for {var}.')
    return float(dayOyear.values)

def annual_scatter(var, obs, hist, fut, db_hist, db_fut, lat, lon, stats = ['median'], save_name = 'annual_scatter', save = False): 
    """
    Create a scatter plot showing the annual cycle of given data over the entire spatial region based on that stat argument passed.
    """
    import math
    logger.info(f'Creating annual scatter plot for {var}.')

    # Adjust plot size to number of stats passed
    if len(stats) > 3:
        # Make additional rows if stats list is too long
        fig, ax = plt.subplots(int(len(stats)/2), math.ceil(len(stats)/2), figsize = (12, 6)) # Round up the number of columns needed to the nearest whole number
        ax = ax.flatten() # Flatten the axes array to make it easier to iterate over
    
    # Keep figure to one row if stats list is 3 items or less
    else:
        fig, ax = plt.subplots(1, len(stats), figsize = (12, 6)) 

    logger.info(f'Obs dataset type: {type(obs)}')
    # Slice dataset to a specific location
    obs_sel = loc_sel(obs, var, lat, lon)
    hist_sel = loc_sel(hist, var, lat, lon)
    fut_sel = loc_sel(fut, var, lat, lon)
    db_hist_sel = loc_sel(db_hist, var, lat, lon)
    db_fut_sel = loc_sel(db_fut, var, lat, lon)
    logger.info(f'Obs dataset type after location selection {type(obs_sel)}')

    # Create list of dates
    dates = pd.date_range(start = '1985-01-01', end = '1985-12-31', freq = 'D') 

    # Create a dictionary to hold a list for each stat
    stat_results = {
        stat: {
            'obs': [],
            'raw_hist': [],
            'raw_fut': [],
            'debiased_hist': [],
            'debiased_fut': []
        } 
        for stat in stats
        } # Dynamically stores data based on whats in stats
    logger.info('Stats dictionary created!')

    for day in dates:
        # Turn day in to usable date string 
        date = day.strftime('%Y-%m-%d')
        logger.info(f'Starting calculations for {date}')

        # Slice dataset down to a given window size centered on date
        # Default window size is 31 days
        obs_window = running_window_slice(obs_sel, date)
        hist_window = running_window_slice(hist_sel, date)
        fut_window = running_window_slice(fut_sel, date)
        db_hist_window = running_window_slice(db_hist_sel, date)
        db_fut_window = running_window_slice(db_fut_sel, date)

        for stat in stats:
            # Calculate the given stat for the sliced dataset
            obs_stat = calc_stat(obs_window, var, stat)
            raw_stat_hist = calc_stat(hist_window, var, stat)
            raw_stat_fut = calc_stat(fut_window, var, stat)
            debiased_stat_hist = calc_stat(db_hist_window, var, stat)
            debiased_stat_fut = calc_stat(db_fut_window, var, stat)

            # Save the result to a list for that stat
            stat_results[stat]['obs'].append(obs_stat)
            stat_results[stat]['raw_hist'].append(raw_stat_hist)
            stat_results[stat]['raw_fut'].append(raw_stat_fut)
            stat_results[stat]['debiased_hist'].append(debiased_stat_hist)
            stat_results[stat]['debiased_fut'].append(debiased_stat_fut)

            logger.info(f'Data stored to {stat} for {date}.')
        
        # Close all datasets out of memory
        obs_window.close()
        hist_window.close()
        fut_window.close()
        db_hist_window.close()
        db_fut_window.close()

    # Create a subplot for each stat
    for i, stat in enumerate(stats):

        # Format dates to include only the month and day
        formatted_dates = list(dates.strftime('%m-%d'))

        # Plot scatter data
        ax[i].plot(
            formatted_dates,
            stat_results[stat]['obs'], 
            'k-',
            alpha = 0.9, 
            markersize = 10,
            label = 'Observations'
        )

        ax[i].plot(
            formatted_dates,
            stat_results[stat]['raw_hist'], 
            'r-.',
            alpha = 0.9, 
            markersize = 10,
            label = 'Historical WRF Output'
        )

        ax[i].plot(
            formatted_dates, 
            stat_results[stat]['raw_fut'], 
            'r-',
            alpha = 0.9,
            markersize = 10,
            label = 'Future WRF Output'
        )

        ax[i].plot(
            formatted_dates,
            stat_results[stat]['debiased_hist'], 
            'g-.',
            alpha = 0.9, 
            markersize = 10,
            label = 'Historical Debiased WRF'
        )

        ax[i].plot(
            formatted_dates,
            stat_results[stat]['debiased_fut'],
            'g-',
            alpha = 0.9,
            markersize = 10,
            label = 'Future Debiased WRF'
        )

        # Adjust plot format settings
        ax[i].set_title(f'{stat}')
        ax[i].set_ylabel(f'{var} ({units[var]})')
        ax[i].grid(True, linestyle = '--', alpha = 0.5)

        # Set x axis labels and tick labels
        ax[i].set_xlabel('Day of Year')
        ax[i].set_xticks(['01-01', '02-01', '03-01', '04-01', '05-01', '06-01', '07-01', '08-01', '09-01', '10-01', '11-01', '12-01'])
        ax[i].tick_params(axis = 'x', rotation = 45)

        # Log what stat was added
        logger.info(f'Scatter plot for {stat} of {var} completed!')

    # Global settings for figure
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper left', bbox_to_anchor = (0.95, 0.95), frameon = True)
    plt.tight_layout()
    plt.show()
    logger.info(f'Annual scatter plot complete for {var}.')

    # Opt to save image to sub directory
    if save:
        stat_str = '_'.join(map(str, stats)) # Pull out items from stat to add to filename
        os.makedirs(current_dir / 'figures' / var, exist_ok = True)
        save_path = current_dir / 'figures' / var / f'{save_name}.png' # Save using actual lat and lon values used
        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Annual scatter plot saved to: {save_path}')

def bias_scatter(var, raw, debiased, save = False):
    """
    Create a scatter plot showing the annual cycle of the bias.
    """
    logger.info(f'Creating bias scatter plot for {var}.')

    # Initialize plot
    fig, ax = plt.subplots(figsize = (12, 6))

    # Check for elevation data output
    ele_path = glob.glob(str(parent_dir / 'wrfout' / 'wrfout*HGT.nc'))
    if not ele_path:
        logger.error('Elevation data not found. Check that elevation_data() saved data to wrfout directory.')
        return
    
    # Open elevation data
    ele_ds = xr.open_dataset(ele_path[0])

    # Calculate the bias
    bias = raw - debiased
    bias = bias.rename({'east_west' : 'west_east'})
    logger.info(f'Initial bias calculations complete for {var}.')
    
    elevation_bands = [[1000, 1500], [1500, 2000], [2000, 2500], [2500, 3000], [3000, 5000]]
    colors = ['b', 'g', 'y', 'orange', 'r']

    for elevations, color in zip(elevation_bands, colors):

        # # Ensure avg_bias and ele_ds are the same shape
        # assert bias['south_north'].shape == ele_ds['south_north'].shape or logger.error('south_north dimensions for bias do not match elevation data.')
        # assert bias['west_east'].shape == ele_ds['west_east'].shape or logger.error('west_east dimensions for bias do not match elevation data.')
        
        # Mask bias data based on elevation range
        mask = (ele_ds['HGT'] >= elevations[0]) & (ele_ds['HGT'] < elevations[1])
        masked_bias = bias.where(mask, drop = True)
        logger.info(f'Bias data masked for elevation range {elevations[0]} to {elevations[1]} m.')
        logger.info(masked_bias)

        # Take the spatial and day of year average of the datasets
        # Average data to get 365 x 1 x 1 (day of year x lat x lon)
        dayOyear = masked_bias[var].groupby('time.dayofyear').mean('time')
        avg_bias = dayOyear.mean(dim = ['lat', 'lon'])

        # Plot scatter data
        ax.plot(
            avg_bias['dayofyear'],
            avg_bias.values, 
            color = color,
            marker = 'o',
            alpha = 0.3, 
            label = f'{elevations[0]} to {elevations[1]} m'
        )

    # Adjust plot format settings
    ax.set_title(f'Mean Annual Cycle of {title[var]} Bias Across Study Region')
    ax.set_ylabel(f'{var} ({units[var]})')
    ax.grid(True, linestyle = '--', alpha = 0.5)
    ax.legend(frameon = True)

    # Set x axis labels and tick labels
    ax.set_xlabel('Day of Year')
    ax.set_xticks(ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366], labels = ['01-01', '02-01', '03-01', '04-01', '05-01', '06-01', '07-01', '08-01', '09-01', '10-01', '11-01', '12-01', ''])
    ax.tick_params(axis = 'x', rotation = 45)

    plt.tight_layout()
    plt.show()
    logger.info(f'Bias scatter plot complete for {var}.')

    # Opt to save image to sub directory
    if save:
        os.makedirs(current_dir / 'figures' / var, exist_ok = True)
        save_path = current_dir / 'figures' / var / f'bias_scatter_{var}.png'
        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Bias scatter plot saved to: {save_path}')

def elevation_data(wrf_output_location):
    """
    Save elevation data for WRF model to a netCDF.
    """
    # Convert path string to pathlib object
    fpath = Path(wrf_output_location)

    # Select first file in output directory
    file = next(fpath.glob('*d03*'))

    with open_or_skip(file) as ds:
        logger.success(f'Successfully opened: {file}')
        
        # Pull elevation data from opened file
        data = ds['HGT']

        # Save elevation data to wrfout directory
        out_path = parent_dir / 'wrfout' / 'wrfout_GSLBIP_multimodel_ssp245_HGT.nc'
        data.squeeze().to_netcdf(out_path)
        logger.success(f'File saved to: {out_path}')

        # Close out of data once saved
        data.close()
        logger.success(f'Elevation data saved!')
    
    return data.squeeze()

def min_n_max_bias(var, fut, debiased_fut, bias_type, save_name = 'min_n_max', save = False):
    """
    2D map of the study region. First panel shows minimum bias at each grid point while second panel shows the 
    maximum. Elevation contours are overlaid on bias data. Note that calculations are only over the future period.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    # TODO: cut into 28-29 years chunks if needed for a better look
    logger.info(f'Creating min and max bias map for {var}.')

    # Find parameters of WRF grid
    proj_lat = fut.attrs.get('CEN_LAT')
    proj_lon = fut.attrs.get('CEN_LON')
    true_lat1 = fut.attrs.get('TRUELAT1')
    true_lat2 = fut.attrs.get('TRUELAT2')

    # Define the Cartopy projection that matches the WRF grid
    wrf_proj = ccrs.LambertConformal(
        central_longitude = proj_lon, 
        central_latitude = proj_lat, 
        standard_parallels = (true_lat1, true_lat2)
    )

    # Initialize plot
    fig, ax = plt.subplots(1, 2, figsize = (16, 9), subplot_kw = {'projection': wrf_proj})
    ax = ax.flatten()

    # Check for elevation data output
    ele_path = glob.glob(str(parent_dir / 'wrfout' / 'wrfout*HGT.nc'))
    if not ele_path:
        logger.error('Elevation data not found. Check that elevation_data() saved data to wrfout directory.')

    # Open elevation data
    ele_ds = xr.open_dataset(ele_path[0])

    # Calculate first the bias of the dataset then the min and max bias
    if bias_type == 'multiplicative':
        bias = fut / debiased_fut
    else:
        bias = fut - debiased_fut

    min_all = bias.min(dim = 'time')
    max_all = bias.max(dim = 'time')

    # Pull out lat and lon values
    lons = fut['lon'].values[0]
    lats = fut['lat'].values[0]

    # Plot data
    min_plot = ax[0].contourf(
        lons, lats,
        min_all[var].values,
        transform = wrf_proj, 
        cmap = cmap('MPL_GnBu', revBool = True),
        shading = 'nearest'
    )

    max_plot = ax[1].contourf(
        lons, lats, 
        max_all[var].values,
        transform = wrf_proj,
        cmap = cmap('MPL_afmhot', revBool = True),
        shading = 'nearest'
        )

    # Add colorbars
    fig.colorbar(min_plot, ax = ax[0], orientation = 'vertical', pad = 0.05, label = f'{var} ({units[var]})', shrink = 0.5)
    fig.colorbar(max_plot, ax = ax[1], orientation = 'vertical', pad = 0.05, label = f'{var} ({units[var]})', shrink = 0.5)

    for i, letter in zip([0,1], ['a.', 'b.']):
        # Add elevation contour lines   
        ele_data = ax[i].contour(lons,
                        lats, 
                        ele_ds['HGT'].values, 
                        levels = [1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000], 
                        colors = 'k', 
                        linewidths = 0.5, 
                        transform = wrf_proj)
        ax[i].clabel(ele_data, inline = True, fmt = '%1.0f', fontsize = 5)

        # Add letter labels to sub plots
        ax[i].text(0.02, 0.93, letter, fontsize = 20, transform = ax[i].transAxes)

    logger.success('Plots completed!')

    # Opt to save image to sub directory
    if save:
        os.makedirs(current_dir / 'figures' / var, exist_ok = True)
        save_path = current_dir / 'figures' / var / f'{save_name}.png'
        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Bias map saved to: {save_path}')

def seasonal_bias(var, fut, debiased_fut, bias_type, save_name = f'seasonal_bias', save = False):
    """
    Creates a map of averaged monthly biases. Each subplot corresponds with a season.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    logger.info(f'Creating seasonal bias map for {var}.')

    # Find parameters of WRF grid
    proj_lat = fut.attrs.get('CEN_LAT')
    proj_lon = fut.attrs.get('CEN_LON')
    true_lat1 = fut.attrs.get('TRUELAT1')
    true_lat2 = fut.attrs.get('TRUELAT2')

    # Define the Cartopy projection that matches the WRF grid
    wrf_proj = ccrs.LambertConformal(
        central_longitude = proj_lon, 
        central_latitude = proj_lat, 
        standard_parallels = (true_lat1, true_lat2)
    )

    # Initialize plot
    fig, ax = plt.subplots(2, 2, figsize = (14, 11), subplot_kw = {'projection': wrf_proj})
    axes = ax.flatten()

    # Check for elevation data output
    ele_path = glob.glob(str(parent_dir / 'wrfout' / 'wrfout*HGT.nc'))
    if not ele_path:
        logger.error('Elevation data not found. Check that elevation_data() saved data to wrfout directory.')

    # Open elevation data
    ele_ds = xr.open_dataset(ele_path[0])

    # Calculate the bias
    if bias_type == 'multiplicative':
            bias = fut / debiased_fut
    else:
        bias = fut - debiased_fut

    logger.info(bias)
    logger.info(f'Initial bias calculations complete for {var}.')

    # Create list of months to show in plots
    months = [1, 4, 7, 10]
    labels = ['a.', 'b.', 'c.', 'd.']

    for i, month, label in zip(axes, months, labels):
        # Select month from the dataset
        month_sel = bias.sel(time = (bias.time.dt.month == month))

        # Take monthly average
        month_avg = month_sel.mean(dim = 'time')
        # When you take the mean it drops the other coords where lat and lon values are stored
        # Use lat and lon values from month_sel instead

        avg_plt = i.pcolormesh(month_sel[var].lon.values[0],
                        month_sel[var].lat.values[0], 
                        month_avg[var].values,
                        shading = 'nearest',
                        cmap = cmap('MPL_coolwarm'),
                        transform = wrf_proj # Tells cartopy lat/lon values are in degrees
        )

        # Add colorbar for the current subplot
        fig.colorbar(avg_plt, ax = i, orientation = 'vertical', pad = 0.05, label = f'{var} ({units[var]})', shrink = 0.75)

        # Add lakes to maps
        lakes = cfeature.NaturalEarthFeature(category = 'physical', name = 'lakes', scale = '50m', facecolor = 'none', edgecolor = 'k')
        i.add_feature(lakes, linewidth = 1.5)

        # Add elevation contour lines
        ele_data = i.contour(month_sel[var].lon.values[0],
                        month_sel[var].lat.values[0], 
                        ele_ds['HGT'].values, 
                        levels = [1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000], 
                        colors = 'k', 
                        linewidths = 0.5, 
                        transform = wrf_proj)
        i.clabel(ele_data, inline = True, fmt = '%1.0f', fontsize = 5)

        # Add letter labels to sub plots
        i.text(0.02, 0.91, label, fontsize = 20, transform = i.transAxes)

    # Opt to save image to sub directory
    if save:
        os.makedirs(current_dir / 'figures' / var, exist_ok = True)
        save_path = current_dir / 'figures' / var / f'{save_name}.png'
        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Seasonal bias map saved to: {save_path}')

# Catch silent errors and report to log file
@logger.catch 
def main(var, bias_type, wrf_output_location, elevation = False):

    if elevation:
        ele_save = elevation_data(wrf_output_location)

    # Open datasets
    obs_path = glob.glob(str(parent_dir / 'gridMET' / f'*{var}*.nc'))
    obs = xr.open_dataset(obs_path[0], decode_times = True)

    mod_path = glob.glob(str(parent_dir / 'daily' / f'*{var}*.nc'))
    mod = xr.open_dataset(mod_path[0], decode_times = True)

    # Separate model data into historical and future periods
    hist = mod.sel(time = slice('1985-01-01', '2014-12-31'))
    fut = mod.sel(time = slice('2015-01-01', '2099-12-31'))

    debiased_hist_path = glob.glob(str(parent_dir / 'wrfout' / f'*{var}*1985-2014.nc'))
    debiased_hist = xr.open_dataset(debiased_hist_path[0], decode_times = True)

    debiased_fut_path = glob.glob(str(parent_dir / 'wrfout' / f'*{var}*2015-2099.nc'))
    debiased_fut = xr.open_dataset(debiased_fut_path[0], decode_times = True)

    # SLC airport coordinates
    lat = 40.788
    lon = -111.978
    place = 'slc_airport'

    # Alta coordinates
    # lat = 40.58083
    # lon = -111.63722
    # place = 'alta'

    # Ogden coordinates
    # lat = 41.2228
    # lon = -111.9733
    # place = 'ogden'

    # Park city coordinates
    # lat = 40.646061
    # lon = -111.49797
    # place = 'park_city'

    # Provo coordinates
    # lat = 40.233845
    # lon = -111.658531
    # place = 'provo'

    # Plot selections
    # scatter = annual_scatter(var, obs, hist, fut, debiased_hist, debiased_fut, lat, lon, stats = ['median', 0.95, 0.5], save_name = f'annual_scatter_{place}_{var}', save = True) 
    # trend = trend_plt(var, obs, hist, fut, debiased_hist, debiased_fut, '1985-03-05', lat, lon, save_name = f'trend_plot_{place}_{var}', save = True)
    cdf = cdf_plt(var, obs, mod, debiased_hist, debiased_fut, dates = ['2070-01-15', '2070-04-15', '2070-07-15', '2070-10-15'], lat = lat, lon = lon, save_name = f'cdf_{place}_{var}',save = True) 
    # seasonal = seasonal_bias(var, fut, debiased_fut, bias_type, save_name = f'seasonal_bias_{var}', save = True)
    # bias_map = min_n_max_bias(var, fut, debiased_fut, bias_type, save_name = f'min_n_max_{var}', save = True) 
    # bias = bias_scatter(var, raw, debiased, save = True)

# ======================
# ---- Entry Point ----
# ======================

if __name__ == '__main__':
    # Track program time in log files
    start = time.perf_counter()
    logger.info('Beginning execution.')
    logger.info('No chunker.')

    # Only inputs required
    main(
        var = 'pr', 
        bias_type = 'multiplicative',
        wrf_output_location = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014'
    )

    # Report of runtime at completion 
    logger.success(f'Debiasing process completed!')
    logger.info(f'Total runtime: {time.perf_counter() - start:.4f}s')


