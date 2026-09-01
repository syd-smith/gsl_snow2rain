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

def trend_plt(var, obs, raw, debiased, fig, ax):
    """
    Create a graph of raw WRF output data, WRF debiased, and observational data to compare trend.
    """

    # Take the spatial average of each dataset 
    obs_mean = obs.mean(dim = ['lat', 'lon'])
    raw_mean = raw.mean(dim = ['lat', 'lon'])
    debiased_mean = debiased.mean(dim = ['lat', 'lon'])

    # Graph obs data
    ax.plot(
        obs_mean['time'],
        obs_mean[var].values, 
        'k-', 
        alpha = 0.4, 
        label = 'Observation'
    )

    # Graph raw WRF output data
    ax.plot(
        raw_mean['time'], 
        raw_mean[var].values, 
        'r-', 
        alpha = 0.4,
        label = 'Raw WRF Output'
    )

    # Graph debiased WRF data
    ax.plot(
        debiased_mean['time'],
        debiased_mean[var].values, 
        'g-',
        alpha = 0.4, 
        label = 'Debiased WRF'
    )

    # Adjust plot format settings
    ax.set_title(f'Climatological Trend of {var} Averaged Across Study Region')
    ax.set_ylabel(f'{var} ({units[var]})')
    ax.set_xlabel('Time')
    ax.grid(True, linestyle = '--', alpha = 0.5)
    ax.legend(frameon = True)

def sample(data, var):
    """ 
    Take a random sample from the given data that is 5% of its original size. 
    """

    # Flatten the data into a 1D array
    flattened = np.array(data[var].flatten())

    # Set size of sample dataset relative to area size
    sample_size = int(len(flattened) * 0.05)

    # Produce sample array
    rng = np.random.default_rng()
    sample = rng.choice(flattened, size = sample_size)

    return sample

def cdf_plt(var, obs, raw, debiased, fig, ax):
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

    # Opt to save image to sub directory
    if save:
        plt.savefig(current_dir / f'cdf_{var}.png', dpi = 300, bbox_inches = 'tight')

    plt.tight_layout()
    plt.show()

def doy_mean(data, var):

    # Average data to get 365 x 1 x 1 (day of year x lat x lon)
    dayOyear = data[var].groupby('time.dayofyear').mean('time')
    spatial_avg = dayOyear.mean(dim = ['lat', 'lon'])

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

    # Opt to save image to sub directory
    if save:
        plt.savefig(current_dir / f'annual_scatter_{var}.png', dpi = 300, bbox_inches = 'tight')

    plt.tight_layout()
    plt.show()

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

# def elevation_scatter():
    
# Catch silent errors and report to log file
@logger.catch 
def main(var, wrf_output_location, elevation = False):

    if elevation:
        ele_save = elevation_data(wrf_output_location)

    # Open datasets
    obs = xr.open_mfdataset(glob.glob(str(parent_dir / 'gridMET' / var / '*.nc')), combine = 'nested', concat_dim = 'time', preprocess = fix_time_coord).sortby('time')
    raw = xr.open_mfdataset(glob.glob(str(parent_dir/ 'daily' / var / '*.nc')), combine = 'nested', concat_dim = 'time', preprocess = fix_time_coord).sortby('time')
    debiased_path = glob.glob(str(parent_dir / 'debiased' / f'*{var}*.nc'))
    debiased = xr.open_dataset(debiased_path[0])

    # Initialize plot
    fig, ax = plt.subplots(figsize = (12, 6))

    # Test Plots
    scatter = annual_scatter(var, obs, raw, debiased, fig, ax, save = True)
    trend = trend_plt(var, obs, raw, debiased, fig, ax, save = True)
    cdf = cdf_plt(var, obs, raw, debiased, fig, ax, save = True)


# ======================
# ---- Entry Point ----
# ======================

if __name__ == '__main__':
    main(
        var = 'tmmn', 
        wrf_output_location = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014'
    )

