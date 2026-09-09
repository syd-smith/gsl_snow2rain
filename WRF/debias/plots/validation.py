# %%
"""
Author: Sydney Smith
Date Created: August 25, 2026
"""

from datetime import datetime
import glob
from ibicus.evaluate.marginal import calculate_marginal_bias, plot_marginal_bias
from loguru import logger
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from statsmodels.distributions.empirical_distribution import ECDF
import sys
import xarray as xr
import xoak
from zoneinfo import ZoneInfo

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from old.temporal_chunks import open_or_skip, get_fpaths

sys.path.append(str(current_dir))
from spatial_chunks import fix_time_coord

# test = xr.open_dataset('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/wrfout/wrfout_GSLBIP_multimodel_ssp245_tmmx.nc')
# test_indexed = test.set_xindex(['lat', 'lon'], 
#     xr.indexes.NDPointIndex, 
#     tree_adapter_cls = xoak.SklearnGeoBallTreeAdapter)

# %%
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

def trend_plt(var, obs, raw, debiased, save = False):
    """
    Create a graph of raw WRF output data, WRF debiased, and observational data to compare trend.
    """
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
        save_path = current_dir / f'trend_{var}.png'
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

    # Ensure the running window length is an odd number (has to be odd for the given day to sit in the exact middle of the window)
    if window_length % 2 == 0:
        logger.error('Window length must be an odd numer.')
        return

    # Cut running window in half to center it around selected date
    half_window = (window_lenth - 1) / 2
    logger.info(f'Half window size: {half_window}')

    # Select bounds of the running window
    start_month = (data.sel(time = date)['time'].values + pd.Timedelta(days = -half_window)).month
    start_day = (data.sel(time = date)['time'].values + pd.Timedelta(days = -half_window)).day
    stop_month = (data.sel(time = date)['time'].values + pd.Timedelta(days = 15)).month
    stop_day = (data.sel(time = date)['time'].values + pd.Timedelta(days = 15)).day

    # Slice data to running window
    window = data.sel(time = (data.time.dt.month == start_month) & (data.time.dt.day >= start_day) | (data.time.dt.month == stop_month) & (data.time.dt.day <= stop_day)).values
    logger.info(window)
    logger.info(f'Data sliced to running window for {date}.')

    return window

def cdf_plt(var, obs, raw, debiased, dates, lat, lon, save = False):
    """
    Create a plot of the cumulative distribution function for each of the given datasets.
    Data will be reduced to the given locations and dates. Dates must be str(%Y/%m/%d).
    This function was intended to create a four pannel plot relecting four dates passed 
    with one date in each season (a seasonal pannel cdf plot).
    """

    # Adjust plot size to number of dates passed
    if len(dates) > 3:
        # Make additional rows if stats list is too long
        fig, ax = plt.subplots(int(len(dates)/2), math.ceil(len(dates)/2), figsize = (12, 6)) # Round up the number of columns needed to the nearest whole number
        ax = ax.flatten() # Flatten the axes array to make it easier to iterate over
    
    # Keep figure to one row if dates list is 3 items or less
    else:
        fig, ax = plt.subplots(1, len(dates), figsize = (12, 6))

    # Reindex datasets to enable spatial selection
    obs_indexed = obs.set_xindex(['lat', 'lon'], 
        xr.indexes.NDPointIndex, 
        tree_adapter_cls = xoak.SklearnGeoBallTreeAdapter) 
    raw_indexed = raw.set_xindex(['lat', 'lon'], 
        xr.indexes.NDPointIndex, 
        tree_adapter_cls = xoak.SklearnGeoBallTreeAdapter)
    debiased_indexed = debiased.set_xindex(['lat', 'lon'], 
        xr.indexes.NDPointIndex, 
        tree_adapter_cls = xoak.SklearnGeoBallTreeAdapter)

    # Slice dataset to a specific location
    obs_sel = obs_indexed.sel(lon = lon, lat = lat, method = 'nearest')
    raw_sel = raw_indexed.sel(lon = lon, lat = lat, method = 'nearest')
    debiased_sel = debiased_indexed.sel(lon = lon, lat = lat, method = 'nearest')

    for date in dates:
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
        obs_cdf = ECDF(obs_window)
        raw_hist_cdf = ECDF(raw_window_hist)
        raw_fut_cdf = ECDF(raw_window_fut)
        debiased_hist_cdf = ECDF(debiased_window_hist)
        debiased_fut_cdf = ECDF(debiased_window_fut)
        logger.info(f'CDFs calculated for {date}.')

        # Plot CDFs
        ax.plot(
            obs_cdf.x, 
            obs_cdf.y, 
            'k-',
            label = 'Observations'
        )

        ax.plot(
            raw_hist_cdf.x,
            raw_hist_cdf.y,
            'ro',
            label = 'Historical WRF Output'
        )

        ax.plot(
            raw_fut_cdf.x, 
            raw_fut_cdf.y, 
            'r-',
            label = 'Future WRF Output'
        )

        ax.plot(
            debiased_hist_cdf.x,
            debiased_fut_cdf.y,
            'go',
            label = 'Historical Debiased WRF'
        )

        ax.plot(
            debiased_cdf.x, 
            debiased_cdf.y, 
            'g-',
            label = 'Future Debiased WRF'
        )

        # Adjust plot format settings for each subplot
        ax.set_title(date)
        ax.set_ylabel('Percentile')
        ax.set_xlabel(f'{var} ({units[var]})')
        ax.grid(True, linestyle = '--', alpha = 0.5)
        ax.legend(frameon = True)

    # Global plot settings
    fig.suptitle(f'CDFs of {title[var]} Data - {window_size} Day Window')
    plt.tight_layout()
    plt.show()
    logger.info(f'CDF plot complete for {var}.')

    # Opt to save image to sub directory
    if save:
        date_str = '_'.join(map(str, dates)) # Pull out items from dates to add to filename
        save_path = current_dir / f'cdf_{var}_{date_str}.png'
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
        dayOyear = data[var].quantile(q_val, dim = 'time')
        logger.success(f'Quantile of {q_val} successfully calculated!')

    # Check for mean or median if stat specificed is not a float
    except (ValueError, TypeError):
        if stat == 'mean':
            dayOyear = data[var].mean('time')
            logger.success('Mean of data successfully calculated!')

        elif stat == 'median':
            dayOyear = data[var].median('time')
            logger.success('Median of data successfully calculated!')

        else:
            logger.error(f'{stat} is not a valid statistic. Please choose from mean, median, or a float between 0 and 1.')

    logger.info(f'Day of year {stat} complete for {var}.')
    return spatial_avg

def annual_scatter(var, obs, raw, debiased, stats = ['median'], save = False): 
    """
    Create a scatter plot showing the annual cycle of given data over the entire spatial region based on that stat argument passed.
    """
    # Adjust plot size to number of stats passed
    if len(stat) > 3:
        # Make additional rows if stats list is too long
        fig, ax = plt.subplots(int(len(stat)/2), math.ceil(len(stat)/2), figsize = (12, 6)) # Round up the number of columns needed to the nearest whole number
        ax = ax.flatten() # Flatten the axes array to make it easier to iterate over
    
    # Keep figure to one row if stats list is 3 items or less
    else:
        fig, ax = plt.subplots(1, len(stat), figsize = (12, 6)) 

    # Slice data to a specfic location
    obs = obs.sel(lon = lon, lat = lat, method = 'nearest')
    raw = raw.sel(lon = lon, lat = lat, method = 'nearest')
    debiased = debiased.sel(lon = lon, lat =lat, method = 'nearest')

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

    for day in dates:
        # Turn day in to usable date string 
        date = day.strftime('%Y-%m-%d')

        # Slice dataset down to a given window size centered on date
        # Default window size is 31 days
        obs_window = running_window_slice(obs, date)
        raw_window = running_window_slice(raw, date)
        debiased_window = running_window_slice(debiased, date)

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

    # Create a subplot for each stat
    for position, stat in enumerate(stats):
        # Log data type of input data
        logger.info(f'Obs datatype: {type(obs)}')
        logger.info(f'Raw datatype: {type(raw)}')
        logger.info(f'Debiased datatype: {type(debiased)})')

        # Format dates to include only the month and day
        formatted_dates = dates.strftime('%m-%d')

        # Plot scatter data
        ax[position].plot(
            formatted_dates,
            stat_results[stat]['obs'], 
            'ko',
            alpha = 0.3, 
            markersize = 10,
            label = 'Observations'
        )

        ax[position].plot(
            formatted_dates,
            stat_results[stat]['raw_hist'], 
            'ro',
            alpha = 0.9, 
            markersize = 3,
            label = 'Historical WRF Output'
        )

        ax[position].plot(
            formatted_dates, 
            stat_results[stat]['raw_fut'], 
            'ro',
            alpha = 0.3,
            markersize = 10,
            label = 'Future WRF Output'
        )

        ax[position].plot(
            formatted_dates,
            stat_results[stat]['debiased_hist'], 
            'go',
            alpha = 0.9, 
            markersize = 3,
            label = 'Historical Debiased WRF'
        )

        ax[position].plot(
            formatted_dates,
            stat_results[stat]['debiased_fut'],
            'go',
            alpha = 0.3,
            markersize = 10,
            label = 'Future Debiased WRF'
        )

        # Adjust plot format settings
        ax[position].set_title(f'{stat}')
        ax[position].set_ylabel(f'{var} ({units[var]})')
        ax[position].grid(True, linestyle = '--', alpha = 0.5)
        ax[position].legend(frameon = True)

        # Set x axis labels and tick labels
        ax[position].set_xlabel('Day of Year')
        ax[position].set_xticks(labels = ['01-01', '02-01', '03-01', '04-01', '05-01', '06-01', '07-01', '08-01', '09-01', '10-01', '11-01', '12-01', ''])
        ax[position].tick_params(axis = 'x', rotation = 45)

        # Log what stat was added
        logger.info(f'Scatter plot for {stat} of {var} completed!')

    # Global settings for figure
    fig.suptitle(f'Mean Annual Cycle of {title[var]} Across Study Region')
    plt.tight_layout()
    plt.show()
    logger.info(f'Annual scatter plot complete for {var}.')

    # Opt to save image to sub directory
    if save:
        stat_str = '_'.join(map(str, stats)) # Pull out items from stat to add to filename
        save_path = current_dir / f'annual_scatter_{var}_{stat_str}.png'
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Annual scatter plot saved to: {save_path}')

def bias_scatter(var, raw, debiased, save = False):
    """
    Create a scatter plot showing the annual cycle of the bias.
    """
    # Initialize plot
    fig, ax = plt.subplots(figsize = (12, 6))

    # Check for elevation data output
    ele_path = glob.glob(str(parent_dir / 'wrfout' / 'wrfout*HGT.nc'))
    if not ele_path:
        logger.error('Elevation data not found. Check that elevation_data() saved data to wrfout directory.')
        return
    
    # Open elevation data
    ele_ds = xr.open_dataset(ele_path[0])

    # Only consider the future period of the datasets
    raw = raw.sel(time = slice('2015-01-01', '2099-12-31'))

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
        avg_bias = doy_stat(masked_bias, var)

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
        save_path = current_dir / f'bias_scatter_{var}.png'
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

def marginal_bias(var, obs, raw, debiased, save = False):
    """
    Create a box and whisker plot of the marginal bias in the debiased dataset.
    """

    # Convert datasets into 1D numpy arrays
    obs = obs[var].to_numpy()
    raw = raw[var].to_numpy()
    debiased = debiased[var].to_numpy()

    # Calculate marginal bias using ibicus evaluate
    calc = calculate_marginal_bias(obs = obs, raw = raw, EDCDF = debiased) # Default statistics are mean, 0.05, 0.95
    logger.info(f'Marginal bias calculations complete for {var}.')

    # Create a box and whisker plot of the marginal bias at specified quantiles
    box_plot = plot_marginal_bias(
        variable = var, 
        bias_df = calc, 
        manual_title = f'Marginal Bias of {title[var]} ({units[var]})'
        ) # Can add statistics title

    logger.info(f'Marginal bias plot for {var} complete.')
    logger.info(type(marg_plot))

    # Create a 2D map of the spatial distribution of the marginal bias
    spatial_plot = plot_bias_spatial(
        variable = var,
        metric = ['mean', 0.05, 0.95],
        bias_df = calc, 
        manual_title = f'Spatial Distribution of Marginal Bias for {title[var]} ({units[var]})'
    )

    # Opt to save images to sub directory
    if save:
        box_path = current_dir / f'marginal_box_plot_{var}.png'
        map_path = current_dir / f'marginal_spatial_plot_{var}.png'

        box_plot.savefig(box_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Marginal bias box plot saved to: {box_path}')

        spatial_plot.savefig(map_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Marginal bias spatial plot saved to: {map_path}')

    # TODO: Save plot to current_dir
    
# Catch silent errors and report to log file
@logger.catch 
def main(var, wrf_output_location, elevation = False):

    if elevation:
        ele_save = elevation_data(wrf_output_location)

    # Open datasets
    obs = xr.open_mfdataset(glob.glob(str(parent_dir / 'gridMET' / var / '*.nc')), combine = 'nested', concat_dim = 'time', preprocess = fix_time_coord).sortby('time')
    raw = xr.open_mfdataset(glob.glob(str(parent_dir/ 'daily' / var / '*.nc')), combine = 'nested', concat_dim = 'time', preprocess = fix_time_coord).sortby('time')
    debiased_path = glob.glob(str(parent_dir / 'wrfout' / f'*{var}*.nc'))
    debiased = xr.open_dataset(debiased_path[0])

    # Test Plots
    scatter = annual_scatter(var, obs, raw, debiased, stat = ['median', 0.95, 0.5], save = True)
    # trend = trend_plt(var, obs, raw, debiased, save = True)

    # # CDF plots for SLC Airport
    # cdf = cdf_plt(var, obs, raw, debiased, dates = ['1985-01-15', '1985-03-15', '1985-06-15', '1985-10-15'], lat = 40.788, lon = -111.978, save = True)
    
    # bias = bias_scatter(var, raw, debiased, save = True)
    # marg = marginal_bias(var, obs, raw, debiased, save = True)

# ======================
# ---- Entry Point ----
# ======================

if __name__ == '__main__':
    main(
        var = 'tmmn', 
        wrf_output_location = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014'
    )

