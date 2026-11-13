"""
Author: Sydney Smith
Date Created: November 12, 2026
"""

from datetime import datetime
import glob
from loguru import logger
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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
from preprocessing import fix_time_coord
from qdm_man import running_window_slice, loc_sel

sys.path.append(str(current_dir))
from validation import best_fit, cdf_masked

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

# =====================
# ---- Functions ----
# =====================

def open_datasets(var):
    """
    Open all necessary datasets for given variable.
    """

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

    return obs, hist, fut, debiased_hist, debiased_fut

# three panel plot tmmx april (monthly mean or specific date)
# raw model -> debiased -> anomaly

# two panel plot of tmmn and tmmx trends to show how T2 was taken to min and max temp
def trend_plt(save_name = 'trend_plot', save = False):
    """
    Create a graph of model WRF output data, observational data, and both detrended datasets to verify detrending 
    was successful. Data plotted is from a single location (SLC Aiport) and a 31 day running window centered around 
    a given date. Plot appear for both min and max temperature.
    """

    # Initialize plot
    fig, ax = plt.subplots(1, 2, figsize = (12, 12), sharex = True)
    axes = ax.flatten()

    # SLC airport coordinates
    lat = 40.788
    lon = -111.978

    for i, var in enumerate(['tmmx', 'tmmn']):
        logger.info(f'Creating detrending verification plot for {var}.')

        obs, hist, fut, db_hist, db_fut = open_datasets(var)

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
        ax[i].plot(
            obs_yearly.time,
            obs_yearly.values, 
            'k-', 
            alpha = 0.6, 
            label = 'Obs'
        )

        ax[i].plot(
                obs_yearly.time,
                best_fit(var, obs_yearly), 
                'k-', 
                alpha = 0.9, 
            )

        # Graph raw WRF output data
        ax[i].plot(
            hist_yearly.time, 
            hist_yearly.values, 
            'r-', 
            alpha = 0.3,
            label = 'WRF'
        )

        ax[i].plot(
            hist_yearly.time,
            best_fit(var, hist_yearly), 
            'r-', 
            alpha = 0.9, 
        )

        ax[i].plot(
            fut_yearly.time, 
            fut_yearly.values, 
            'r-', 
            alpha = 0.3,
            label = 'WRF'
        )

        ax[i].plot(
            fut_yearly.time,
            best_fit(var, fut_yearly), 
            'r-', 
            alpha = 0.9, 
        )

        # Graph debiased data
        ax[i].plot(
            db_hist_yearly.time,
            db_hist_yearly.values, 
            'g-',
            alpha = 0.3, 
            label = 'Detrended Obs'
        )

        ax[i].plot(
            db_hist_yearly.time,
            best_fit(var, db_hist_yearly), 
            'g-', 
            alpha = 0.9, 
        )

        ax[i].plot(
                db_fut_yearly.time,
                db_fut_yearly.values, 
                'g-',
                alpha = 0.3, 
                label = 'Detrended Obs'
            )
        
        ax[i].plot(
            db_fut_yearly.time,
            best_fit(var, db_fut_yearly),
            'g-', 
            alpha = 0.9, 
        )

        # Adjust plot format settings
        # ax.set_title(f'Trended vs. Detrended {title[var]} Data')
        ax[i].set_ylabel(f'{var} ({units[var]})')
        ax[i].set_xlabel('Time')
        ax[i].grid(True, linestyle = '--', alpha = 0.5)

        # Set labels for the x axis
        ticks = [f'{yr}-01-01T00:00:00.000000000' for yr in range(1985, 2100, 5)]
        labels = [f'{yr}' for yr in range(1985, 2100, 5)]
        ax[i].set_xticks(ticks = ticks, labels = labels)
        ax[i].tick_params(axis = 'x', rotation = 45)

        logger.info(f'Trend plot complete for {var}.')

    plt.legend(frameon = True)
    plt.tight_layout()
    plt.show()
    

    # Opt to save image to sub directory
    if save:
        os.makedirs(current_dir / 'figures' / var, exist_ok = True)
        save_path = current_dir / 'figures' / var / f'{save_name}.png'
        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Trend plot saved to: {save_path}')

# CDF plt of srad, uas, tmmn, and pr at slc airport for june 15
def cdf_plt(save_name, save = True):
    """
    Create a plot of the cumulative distribution function for each of the given datasets.
    Data will be reduced to the given locations and dates. Dates must be str(%Y/%m/%d).
    This function was intended to create a four pannel plot relecting four dates passed 
    with one date in each season (a seasonal pannel cdf plot).
    """
    import math
    from statsmodels.distributions.empirical_distribution import ECDF
    logger.info(f'Creating cdf plot for {var}.')

    # Adjust plot size to number of variables
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    ax = ax.flatten() # Flatten the axes array to make it easier to iterate over

    # Set date of interest to June 15
    date = pd.to_datetime('1988-06-15')

    # SLC airport coordinates
    lat = 40.788
    lon = -111.978

    # List of plot labels and variables
    labels = ['a.', 'b.', 'c.', 'd.']
    vars = ['tmmn', 'pr', 'srad', 'uas']

    for i, (var, label) in enumerate(zip(vars, labels)):
        # Open datasets for given variable
        obs, mod, db_hist, db_fut = open_datasets(var)

        # Slice dataset to a specific location
        obs_sel = loc_sel(obs, var, lat, lon)
        mod_sel = loc_sel(mod, var, lat, lon)
        db_hist_sel = loc_sel(db_hist, var, lat, lon)
        db_fut_sel = loc_sel(db_fut, var, lat, lon)

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
        ax[i].text(15, 0.02, label, fontsize = 15)
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

# sph annual scatter plot

def mean_bias_map(var, save_name = 'mean_bias_map', save = False):
    """
    Creates a map of averaged biases. Elevational variations are meant to pop out. Break into seasonal if more clarity is needed.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    logger.info(f'Creating mean bias map for {var}.')

    # open datasets
    obs, hist, fut, debiased_hist, debiased_fut = open_datasets(var)
    
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
    fig, ax = plt.subplots(figsize = (10, 10), subplot_kw = {'projection': wrf_proj})

    # Check for elevation data output
    ele_path = glob.glob(str(parent_dir / 'wrfout' / 'wrfout*HGT.nc'))
    if not ele_path:
        logger.error('Elevation data not found. Check that elevation_data() saved data to wrfout directory.')

    # Open elevation data
    ele_ds = xr.open_dataset(ele_path[0])

    # Calculate the bias (meant to handle additive bias calculation for tmmn)
    bias = fut - debiased_fut
    logger.info(bias)
    logger.info(f'Initial bias calculations complete for {var}.')

    # Take monthly average
    bias_mean = bias.mean(dim = 'time')
    # When you take the mean it drops the other coords where lat and lon values are stored
    # Use lat and lon values from fut instead

    avg_plt = ax.pcolormesh(fut[var].lon.values[0],
                    fut[var].lat.values[0], 
                    bias_mean[var].values,
                    shading = 'nearest',
                    cmap = cmap('MPL_coolwarm'),
                    transform = wrf_proj # Tells cartopy lat/lon values are in degrees
    )

    # Add colorbar centered on zero
    norm = mcolors.TwoSlopeNorm(vmin = -2, vmax = 8, vcenter = 0)
    fig.colorbar(avg_plt, ax = ax, norm = norm, orientation = 'vertical', pad = 0.05, label = f'{var} ({units[var]})', shrink = 0.75)

    # Add elevation contour lines
    ele_data = ax.contour(fut[var].lon.values[0],
                    fut[var].lat.values[0], 
                    ele_ds['HGT'].values, 
                    levels = [1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000], 
                    colors = 'k', 
                    linewidths = 0.5, 
                    transform = wrf_proj)
    ax.clabel(ele_data, inline = True, fmt = '%1.0f', fontsize = 5)

    # Opt to save image to sub directory
    if save:
        os.makedirs(current_dir / 'figures', exist_ok = True)
        save_path = current_dir / 'figures' / f'{save_name}.png'
        fig.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Seasonal bias map saved to: {save_path}')



