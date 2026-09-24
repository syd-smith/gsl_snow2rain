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
from spatial_chunks import fix_time_coord

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

def loc_sel(ds, lat, lon):
    """
    Select the nearest point to the lat and lon dims passed.
    """
    logger.info(f'Selecting nearest location to {lat}, {lon}.')

    # Select single array along time dim to reduce ds size
    lat_2d = ds['lat'].isel(time = 0)
    lon_2d = ds['lon'].isel(time = 0)

    # Find the vector distance from the coordinates passed to every coordinate pair on the dataset's grid
    dist = (lat_2d - (lat)) ** 2 + (lon_2d - (lon)) ** 2

    # Find the min distance and pull out its x and y index values
    dist_min = dist.argmin()
    logger.info(f'Point selected is {dist_min} from {lat}, {lon}.')
    y_idx, x_idx = np.unravel_index(dist_min, dist.shape)
    logger.info(f'Min distance is located at {x_idx}, {y_idx}.')
    
    # SPull grid's spatial dims dynamically
    spatial_dims = ds['lat'].dims

    # Select min distance grid point using isel
    point_data = ds.isel({
        spatial_dims[1]: y_idx,
        spatial_dims[2]: x_idx
    }).squeeze()

    logger.info(f'New lat: {float(point_data["lat"][0])}')
    logger.info(f'New lon: {float(point_data["lon"][0])}')

    return point_data

def trend_plt(var, obs, raw, debiased, save = False):
    """
    Create a graph of raw WRF output data, WRF debiased, and observational data to compare trend.
    """
    logger.info(f'Creating trend plot for {var}.')

    # Initialize plot
    fig, ax = plt.subplots(figsize = (12, 6))

    # Take the spatial average of each dataset 
    obs_mean = obs.mean(dim = ['lat', 'lon'])
    raw_mean = raw.mean(dim = ['lat', 'lon'])
    debiased_mean = debiased.mean(dim = ['lat', 'lon'])

    # Resample the data to yearly averages for trend calculations
    obs_yearly = obs_mean[var].resample(time = '1YS').mean()
    raw_yearly = raw_mean[var].resample(time = '1YS').mean()
    debiased_yearly = debiased_mean[var].resample(time = '1YS').mean()
    
    # Log metadata on yearly averages for debugging
    logger.info(obs_yearly)
    logger.info(obs_yearly.time)
    logger.info(f'Averages complete for {var} trend calculations.')
    
    # Graph obs data
    ax.plot(
        obs_yearly.time,
        obs_yearly.values, 
        'k-', 
        alpha = 0.9, 
        label = 'Observation'
    )

    # Graph raw WRF output data
    ax.plot(
        raw_yearly.time, 
        raw_yearly.values, 
        'r-', 
        alpha = 0.9,
        label = 'Raw WRF Output'
    )

    # Graph debiased WRF data
    ax.plot(
        debiased_yearly.time,
        debiased_yearly.values, 
        'g-',
        alpha = 0.9, 
        label = 'Debiased WRF'
    )

    # Adjust plot format settings
    ax.set_title(f'Climatological Trend of {title[var]} Averaged Across Study Region')
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
        save_path = current_dir / 'figures' / f'trend_{var}.png'
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
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

def running_window_slice(data, date, window_length = 31):
    """"
    Select only data within the window length (measured in number of days and centered around
    date given) across the entire dataset.
    """
    # Ensure the running window length is an odd number (has to be odd for the given day to sit in the exact middle of the window)
    if window_length % 2 == 0:
        logger.error('Window length must be an odd numer.')
        return

    # Cut running window in half to center it around selected date
    half_window = (window_length - 1) / 2
    logger.info(f'Half window size: {half_window}')

    # Select bounds of the running window
    start_month = (data.sel(time = date, method = 'nearest')['time'].values + pd.Timedelta(days = -half_window)).month
    start_day = (data.sel(time = date, method = 'nearest')['time'].values + pd.Timedelta(days = -half_window)).day
    stop_month = (data.sel(time = date, method = 'nearest')['time'].values + pd.Timedelta(days = half_window)).month
    stop_day = (data.sel(time = date, method = 'nearest')['time'].values + pd.Timedelta(days = half_window)).day

    # Slice data to running window
    window = data.sel(time = (data.time.dt.month == start_month) & (data.time.dt.day >= start_day) | (data.time.dt.month == stop_month) & (data.time.dt.day <= stop_day))
    logger.info(window)
    logger.info(f'Data sliced to running window for {date}.')

    return window

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

def cdf_plt(var, obs, raw, debiased, dates, lat, lon, save = False):
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
        fig, ax = plt.subplots(int(len(dates)/2), math.ceil(len(dates)/2), figsize = (12, 6)) # Round up the number of columns needed to the nearest whole number
        ax = ax.flatten() # Flatten the axes array to make it easier to iterate over
    
    # Keep figure to one row if dates list is 3 items or less
    else:
        fig, ax = plt.subplots(1, len(dates), figsize = (12, 6))

    # Slice dataset to a specific location
    obs_sel = loc_sel(obs, lat, lon)
    raw_sel = loc_sel(raw, lat, lon)
    debiased_sel = loc_sel(debiased, lat, lon)

    for i, date in enumerate(dates):
        # Slice dataset down to a given window size centered on date
        # Default window size is 31 days
        window_size = 31
        obs_window = running_window_slice(obs_sel, date, window_size)
        raw_window = running_window_slice(raw_sel, date, window_size)
        debiased_window = running_window_slice(debiased_sel, date, window_size)

        # Slice into historical and future periods
        raw_window_hist = raw_window.sel(time = raw_window.time.dt.year.isin(range(1985, 2015)))
        raw_window_fut = raw_window.sel(time = raw_window.time.dt.year.isin(range(2015, 2100)))
        debiased_window_hist = debiased_window.sel(time = debiased_window.time.dt.year.isin(range(1985, 2015)))
        debiased_window_fut = debiased_window.sel(time = debiased_window.time.dt.year.isin(range(2015, 2100)))

        # Calculate a CDF for each array
        obs_cdf = ECDF(obs_window[var].values.flatten())
        raw_hist_cdf = ECDF(raw_window_hist[var].values.flatten())
        raw_fut_cdf = ECDF(raw_window_fut[var].values.flatten())
        debiased_hist_cdf = ECDF(debiased_window_hist[var].values.flatten())
        debiased_fut_cdf = ECDF(debiased_window_fut[var].values.flatten())
        logger.info(f'CDFs calculated for {date}.')

        # Mask CDF outputs
        obs_x, obs_y = cdf_masked(obs_cdf)
        raw_hist_x, raw_hist_y = cdf_masked(raw_hist_cdf)
        raw_fut_x, raw_fut_y = cdf_masked(raw_fut_cdf)
        debiased_hist_x, debiased_hist_y = cdf_masked(debiased_hist_cdf)
        debiased_fut_x, debiased_fut_y = cdf_masked(debiased_fut_cdf)

        # Plot CDFs
        ax[i].plot(
            obs_x, 
            obs_y, 
            'k-',
            label = 'Observations'
        )

        ax[i].plot(
            raw_hist_x,
            raw_hist_y,
            'r-.',
            label = 'Historical WRF Output'
        )

        ax[i].plot(
            raw_fut_x, 
            raw_fut_y, 
            'r-',
            label = 'Future WRF Output'
        )

        ax[i].plot(
            debiased_hist_x,
            debiased_hist_y,
            'g-.',
            label = 'Historical Debiased WRF'
        )

        ax[i].plot(
            debiased_fut_x, 
            debiased_fut_y, 
            'g-',
            label = 'Future Debiased WRF'
        )

        # Adjust plot format settings for each subplot
        ax[i].set_title(date)
        ax[i].set_ylabel('Percentile')
        ax[i].set_xlabel(f'{var} ({units[var]})')
        ax[i].grid(True, linestyle = '--', alpha = 0.5)
        logger.info(f'Plotting completed for {date}.')

    # Global plot settings
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'upper left', bbox_to_anchor = (0.95, 0.95), frameon = True)
    plt.tight_layout()
    plt.show()
    logger.info(f'CDF plot complete for {var}.')

    # Opt to save image to sub directory
    if save:
        date_str = '_'.join(map(str, dates)) # Pull out items from dates to add to filename
        save_path = current_dir / 'figures' / f'cdf_{var}_{date_str}_lat:{float(debiased_sel["lat"][0]):.2f}_lon:{float(debiased_sel["lon"][0]):.2f}.png' # Pull actual grid points used
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'CDF plot saved to: {save_path}')

def calc_stat(data, var, stat = 'mean'):
    """"
    Average data based on the specified statistic.
    """
    logger.info(type(data))

    # Group the data by day of year and calculate the specified statistic (mean, median, 0.05, or 0.95) for each day of the year
    try:
        q_val = float(stat)
        logger.info(f'Quantile value of {q_val} selected for {var}.')

        # Apply quantile function
        dayOyear = data[var].quantile(q_val)
        logger.success(f'Quantile of {q_val} successfully calculated!')

    # Check for mean or median if stat specificed is not a float
    except (ValueError, TypeError):
        if stat == 'mean':
            dayOyear = data[var].mean()
            logger.success('Mean of data successfully calculated!')

        elif stat == 'median':
            dayOyear = data[var].median()
            logger.success('Median of data successfully calculated!')

        else:
            logger.error(f'{stat} is not a valid statistic. Please choose from mean, median, or a float between 0 and 1.')

    logger.info(f'calc_stat returns {float(dayOyear.values)}.')
    logger.info(f'Day of year {stat} complete for {var}.')
    return float(dayOyear.values)

def annual_scatter(var, obs, raw, debiased, lat, lon, stats = ['median'], save = False): 
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
    obs_sel = loc_sel(obs, lat, lon)
    raw_sel = loc_sel(raw, lat, lon)
    debiased_sel = loc_sel(debiased, lat, lon)
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
        raw_window = running_window_slice(raw_sel, date)
        debiased_window = running_window_slice(debiased_sel, date)

        # Slice into historical and future periods
        raw_window_hist = raw_window.sel(time = raw_window.time.dt.year.isin(range(1985, 2015)))
        raw_window_fut = raw_window.sel(time = raw_window.time.dt.year.isin(range(2015, 2100)))
        debiased_window_hist = debiased_window.sel(time = debiased_window.time.dt.year.isin(range(1985, 2015)))
        debiased_window_fut = debiased_window.sel(time = debiased_window.time.dt.year.isin(range(2015, 2100)))

        for stat in stats:
            # Calculate the given stat for the sliced dataset
            obs_stat = calc_stat(obs_window, var, stat)
            raw_stat_hist = calc_stat(raw_window_hist, var, stat)
            raw_stat_fut = calc_stat(raw_window_fut, var, stat)
            debiased_stat_hist = calc_stat(debiased_window_hist, var, stat)
            debiased_stat_fut = calc_stat(debiased_window_fut, var, stat)

            # Save the result to a list for that stat
            stat_results[stat]['obs'].append(obs_stat)
            stat_results[stat]['raw_hist'].append(raw_stat_hist)
            stat_results[stat]['raw_fut'].append(raw_stat_fut)
            stat_results[stat]['debiased_hist'].append(debiased_stat_hist)
            stat_results[stat]['debiased_fut'].append(debiased_stat_fut)

            logger.info(f'Data stored to {stat} for {date}.')
        
        # Close all datasets out of memory
        obs_window.close()
        raw_window.close()
        debiased_window.close()
        raw_window_hist.close()
        raw_window_fut.close()
        debiased_window_hist.close()
        debiased_window_fut.close()

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
        save_path = current_dir / 'figures' / f'annual_scatter_{var}_{stat_str}_lat:{float(debiased_sel["lat"][0]):.2f}_lon:{float(debiased_sel["lon"][0]):.2f}.png' # Save using actual lat and lon values used
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
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
        save_path = current_dir / 'figures' / f'bias_scatter_{var}.png'
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
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

def min_n_max_bias(var, raw, debiased, save = False):
    """
    2D map of the study region. First panel shows minimum bias at each grid point while second panel shows the 
    maximum. Elevation contours are overlaid on bias data. Note that calculations are only over the future period.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    # TODO: cut into 28-29 years chunks if needed for a better look
    logger.info(f'Creating min and max bias map for {var}.')

    # Find parameters of WRF grid
    proj_lat = debiased.attrs.get('CEN_LAT')
    proj_lon = debiased.attrs.get('CEN_LON')
    true_lat1 = debiased.attrs.get('TRUELAT1')
    true_lat2 = debiased.attrs.get('TRUELAT2')

    # Define the Cartopy projection that matches the WRF grid
    wrf_proj = ccrs.LambertConformal(
        central_longitude = proj_lon, 
        central_latitude = proj_lat, 
        standard_parallels = (true_lat1, true_lat2)
    )

    # Initialize plot
    fig, ax = plt.subplots(1, 2, figsize = (12, 6), subplot_kw = {'projection': wrf_proj})
    ax = ax.flatten()

    # Check for elevation data output
    ele_path = glob.glob(str(parent_dir / 'wrfout' / 'wrfout*HGT.nc'))
    if not ele_path:
        logger.error('Elevation data not found. Check that elevation_data() saved data to wrfout directory.')
        return

    # TODO: add elevation contours

    # Open elevation data
    ele_ds = xr.open_dataset(ele_path[0])

    for year in range(1985, 2100):
        logger.info(f'Calculating extremes for {year}.')
        # Isolate one year of data at a time
        raw_yr = raw.sel(time = slice(f'{year}-01-01', f'{year}-12-31'))
        debiased_yr = debiased.sel(time = slice(f'{year}-01-01', f'{year}-12-31'))

        # Calculate the bias and take the min and max
        bias = raw_yr - debiased_yr
        bias_min = bias.min(dim = 'time')
        bias_max = bias.max(dim = 'time')
        logger.info(f'Min for {year}: {bias_min[var]}')
        logger.info(f'Max for {year}: {bias_max[var]}')

        if year == 1985:
            # Skip combining the first year of data (there's nothing to combine it with)
            min_all = bias_min
            max_all = bias_max

        else:
            # Combine datasets together on a new dim
            min_concat = xr.concat([min_all, bias_min], dim = 'year')
            max_concat = xr.concat([max_all, bias_max], dim = 'year')

            # Take the min of the new combined dataset (this will cause the values that are not most extreme to be left behind)
            min_all = min_concat.min(dim = 'year')
            max_all = max_concat.max(dim = 'year')

        logger.info(f'Overall min: {min_all[var]}')
        logger.info(f'Overall max: {max_all[var]}')

        # Close out of that years worth of data
        bias_min.close()
        bias_max.close()

    # Log once calculations are complete
    logger.info('Min and max calculations complete! Starting plotting.')

    # Pull out lat and lon values
    lons = min_all['lon'].values
    lats = min_all['lat'].values

    # Plot data
    min_plot = ax[0].pcolormesh(
        lons, lats,
        min_all[var].values,
        transform = wrf_proj, 
        cmap = cmap('GMT_ocean', revBool = True),
        shading = 'nearest'
    )

    max_plot = ax[1].pcolormesh(
        lons, lats, 
        max_all[var].values,
        transform = wrf_proj,
        cmap = cmap('MPL_afmhot', revBool = True),
        shading = 'nearest'
        )

    # Add colorbars
    fig.colorbar(min_plot, ax = ax[0], orientation = 'vertical', pad = 0.05, label = f'{var} ({units[var]})', shrink = 0.7)
    fig.colorbar(max_plot, ax = ax[1], orientation = 'vertical', pad = 0.05, label = f'{var} ({units[var]})', shrink = 0.7)

    # Add lakes to maps
    lakes = cfeature.NaturalEarthFeature(category = 'physical', name = 'lakes', scale = '50m', facecolor = 'none', edgecolor = 'k')
    ax[0].add_feature(lakes, linewidth = 1.5)
    ax[1].add_feature(lakes, linewidth = 1.5)

    # Add letter labels to sub plots
    ax[0].text(0.02, 0.88, 'a.', fontsize = 10, transform = ax[0].transAxes)
    ax[1].text(0.02, 0.88, 'b.', fontsize = 10, transform = ax[1].transAxes)

    logger.success('Plots completed!')

    # Opt to save image to sub directory
    if save:
        save_path = current_dir / 'figures' / f'min_max_bias_{var}.png'
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Bias map saved to: {save_path}')

def seasonal_bias(var, raw, debiased, save = False):
    """
    Creates a map of averaged monthly biases. Each subplot corresponds with a season.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    logger.info(f'Creating seasonal bias map for {var}.')

    # Find parameters of WRF grid
    proj_lat = debiased.attrs.get('CEN_LAT')
    proj_lon = debiased.attrs.get('CEN_LON')
    true_lat1 = debiased.attrs.get('TRUELAT1')
    true_lat2 = debiased.attrs.get('TRUELAT2')

    # Define the Cartopy projection that matches the WRF grid
    wrf_proj = ccrs.LambertConformal(
        central_longitude = proj_lon, 
        central_latitude = proj_lat, 
        standard_parallels = (true_lat1, true_lat2)
    )

    # Initialize plot
    fig, ax = plt.subplots(2, 2, figsize = (12, 6), subplot_kw = {'projection': wrf_proj})
    ax = ax.flatten()

    # Check for elevation data output
    ele_path = glob.glob(str(parent_dir / 'wrfout' / 'wrfout*HGT.nc'))
    if not ele_path:
        logger.error('Elevation data not found. Check that elevation_data() saved data to wrfout directory.')
        return

    # Open elevation data
    ele_ds = xr.open_dataset(ele_path[0])

    # Refine data to future period
    raw_fut = raw.sel(time = slice('2015-01-01', '2099-12-31'))
    debiased_fut = debiased.sel(time = slice('2015-01-01', '2099-12-31'))

    # Calculate the bias
    bias = raw_fut - debiased_fut
    bias = bias.rename({'east_west' : 'west_east'})
    logger.info(bias)
    logger.info(f'Initial bias calculations complete for {var}.')

    # Create list of months to show in plots
    months = [1, 4, 7, 10]
    labels = ['a.', 'b.', 'c.', 'd.']
    
    for i, month, label in enumerate(zip(months, labels)):
        # Select month from the dataset
        month_sel = bias.sel(time = (bias.time.dt.month == month))

        # Take monthly average
        month_avg = month_sel.mean(dim = 'time')

        avg_plt = month_avg[var].pcolormesh(
            ax = ax[i],
            x = 'lon',
            y = 'lat',
            shading = 'nearest',
            cmap = cmap('MPL_coolwarm'),
            transform = wrf_proj # Tells cartopy lat/lon values are in degrees
        )

        fig.colorbar(avg_plt, ax = ax[i], orientation = 'vertical', pad = 0.05, label = f'{var} ({units[var]})', shrink = 0.7)

        # Add lakes to maps
        lakes = cfeature.NaturalEarthFeature(category = 'physical', name = 'lakes', scale = '50m', facecolor = 'none', edgecolor = 'k')
        ax[i].add_feature(lakes, linewidth = 1.5)

        # Add letter labels to sub plots
        ax[i].text(0.02, 0.88, label, fontsize = 60, transform = ax[i].transAxes)

    # Opt to save image to sub directory
    if save:
        save_path = current_dir / 'figures' / f'spatial_bias_{var}.png'
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Seasonal bias map saved to: {save_path}')

# Catch silent errors and report to log file
@logger.catch 
def main(var, wrf_output_location, elevation = False):

    if elevation:
        ele_save = elevation_data(wrf_output_location)

    # Open datasets
    obs_path = glob.glob(str(parent_dir / 'gridMET' / f'*{var}*.nc'))
    obs = xr.open_dataset(obs_path[0], decode_times = True)

    raw_path = glob.glob(str(parent_dir / 'daily' / f'*{var}*.nc'))
    raw = xr.open_dataset(raw_path[0], decode_times = True)

    debiased_path = glob.glob(str(parent_dir / 'wrfout' / f'*{var}*.nc'))
    debiased = xr.open_dataset(debiased_path[0], decode_times = True)

    # Set location of interest as SLC airport
    lat = 40.788
    lon = -111.978

    # Test Plots
    # scatter = annual_scatter(var, obs, raw, debiased, lat, lon, stats = ['median', 0.95, 0.5], save = True) 
    # trend = trend_plt(var, obs, raw, debiased, save = True)

    # CDF plots for SLC Airport
    cdf = cdf_plt(var, obs, raw, debiased, dates = ['1985-01-15', '1985-03-15', '1985-06-15', '1985-10-15'], lat = lat, lon = lon, save = True) 
    
    # seasonal = seasonal_bias(var, raw, debiased, save = False)
    # bias_map = min_n_max_bias(var, raw, debiased, save = True) # OOM kill (chunking didn't work might need to try mannual chunking)
    # bias = bias_scatter(var, raw, debiased, save = True)

# ======================
# ---- Entry Point ----
# ======================

# if __name__ == '__main__':
#     main(
#         var = 'tmmx', 
#         wrf_output_location = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014'
#     )

var = 'tmmx'


debiased_path = glob.glob(str(parent_dir / 'wrfout' / f'*{var}*.nc'))
debiased = xr.open_dataset(debiased_path[0], decode_times = True)
print(debiased.min())
