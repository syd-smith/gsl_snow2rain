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
from statsmodels.distributions.empirical_distribution import ECDF
import sys
import xarray as xr
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

def trend_plt(var, obs, raw, debiased, fig, ax, save = False):
    """
    Create a graph of raw WRF output data, WRF debiased, and observational data to compare trend.
    """

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

def cdf_plt(var, obs, raw, debiased, fig, ax, save = False):
    """
    Create a plot of the cumulative distribution function for each of the given datasets.
    """

    # Take a sample of the given datasets that is 5% of its original size
    sample_obs = sample(obs, var)
    sample_raw = sample(raw, var)
    sample_debiased = sample(debiased, var)

    # Calculate a cdf for each array
    obs_cdf = ECDF(sample_obs)
    raw_cdf = ECDF(sample_raw)
    debiased_cdf = ECDF(sample_debiased)
    logger.info(f'CDFs calculated for {var}.')

    # Plot CDFs
    ax.plot(
        obs_cdf.x, 
        obs_cdf.y, 
        'k-',
        label = 'Observations'
    )

    ax.plot(
        raw_cdf.x, 
        raw_cdf.y, 
        'r-',
        label = 'Raw WRF Output'
    )

    ax.plot(
        debiased_cdf.x, 
        debiased_cdf.y, 
        'g-',
        label = 'Debiased WRF'
    )

    # Adjust plot format settings
    ax.set_title(f'CDFs of {title[var]} Sampled Data')
    ax.set_ylabel('Percentile')
    ax.set_xlabel(f'{var} ({units[var]})')
    ax.grid(True, linestyle = '--', alpha = 0.5)
    ax.legend(frameon = True)

    plt.tight_layout()
    plt.show()
    logger.info(f'CDF plot complete for {var}.')

    # Opt to save image to sub directory
    if save:
        save_path = current_dir / f'cdf_{var}.png'
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'CDF plot saved to: {save_path}')

def doy_mean(data, var):

    # Average data to get 365 x 1 x 1 (day of year x lat x lon)
    dayOyear = data[var].groupby('time.dayofyear').mean('time')
    spatial_avg = dayOyear.mean(dim = ['lat', 'lon'])
    logger.info(f'Spatial and day of year averages complete for {var}.')

    return spatial_avg

def annual_scatter(var, obs, raw, debiased, fig, ax, save = False):
    """
    Create a scatter plot showing the annual cycle of given data over the entire spatial region.
    """

    # Take the spatial and day of year average of the datasets
    obs = doy_mean(obs, var)
    raw = doy_mean(raw, var)
    debiased = doy_mean(debiased, var)

    # Plot scatter data
    ax.plot(
        obs['dayofyear'],
        obs.values, 
        'ko',
        alpha = 0.3, 
        label = 'Observations'
    )

    ax.plot(
        raw['dayofyear'],
        raw.values, 
        'ro',
        alpha = 0.3, 
        label = 'Raw WRF Output'
    )

    ax.plot(
        debiased['dayofyear'],
        debiased.values, 
        'go',
        alpha = 0.3, 
        label = 'WRF Debiased'
    )

    # Adjust plot format settings
    ax.set_title(f'Mean Annual Cycle of {title[var]} Across Study Region')
    ax.set_ylabel(f'{var} ({units[var]})')
    ax.grid(True, linestyle = '--', alpha = 0.5)
    ax.legend(frameon = True)

    # Set x axis labels and tick labels
    ax.set_xlabel('Day of Year')
    ax.set_xticks(ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366], labels = ['01-01', '02-01', '03-01', '04-01', '05-01', '06-01', '07-01', '08-01', '09-01', '10-01', '11-01', '12-01', ''])
    ax.tick_params(axis = 'x', rotation = 45)

    plt.tight_layout()
    plt.show()
    logger.info(f'Annual scatter plot complete for {var}.')

    # Opt to save image to sub directory
    if save:
        save_path = current_dir / f'annual_scatter_{var}.png'
        plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        logger.success(f'Annual scatter plot saved to: {save_path}')

def bias_scatter(var, raw, debiased, fig, ax, save = False):
    """
    Create a scatter plot showing the annual cycle of the bias.
    """

    # Check for elevation data output
    ele_path = glob.glob(str(parent_dir / 'wrfout' / 'wrfout*HGT.nc'))
    if not ele_path:
        logger.error('Elevation data not found. Check that elevation_data() saved data to wrfout directory.')
    
    # Open elevation data
    ele_ds = xr.open_dataset(ele_path[0])

    # Only consider the future period of the datasets
    raw = raw.sel(time = slice('2015-01-01', '2099-12-31'))

    # Calculate the bias
    bias = raw - debiased
    logger.info(f'Initial bias calculations complete for {var}.')
    
    elevation_bands = [[1000, 1500], [1500, 2000], [2000, 2500], [2500, 3000], [3000, 5000]]
    colors = ['b', 'g', 'y', 'o', 'r']

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
        avg_bias = doy_mean(bias, var)

        # Plot scatter data
        ax.plot(
            avg_bias['dayofyear'],
            avg_bias.values, 
            f'{color}o',
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

    # Initialize plot
    fig, ax = plt.subplots(figsize = (12, 6))

    # Test Plots
    # scatter = annual_scatter(var, obs, raw, debiased, fig, ax, save = True)
    # trend = trend_plt(var, obs, raw, debiased, fig, ax, save = True)
    # cdf = cdf_plt(var, obs, raw, debiased, fig, ax, save = True)
    bias = bias_scatter(var, raw, debiased, fig, ax, save = True)

# ======================
# ---- Entry Point ----
# ======================

if __name__ == '__main__':
    main(
        var = 'tmmn', 
        wrf_output_location = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014'
    )

