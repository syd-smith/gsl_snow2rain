"""
Author: Sydney Smith
Date Created: October 26, 2026
"""

from contextlib import contextmanager
from datetime import datetime
import glob
from loguru import logger
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
import time
import xarray as xr
import xesmf as xe
from zoneinfo import ZoneInfo

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

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
    'pr': 'pr',
    'sph': 'sph',
    'srad': 'rsds',
    'vas': 'sfcwind',
    'uas': 'sfcwind',
}

# =====================
# ---- Functions ----
# =====================

def get_fpaths(f_loc, domain):
    """
    Input path to directory that contains WRF data. Outputs a sorted list of file names.
    """

    # Convert f_loc to a Path object for clean path joining
    input_dir = Path(f_loc)
    
    # Use recursive glob '**' to search inside all subdirectories for files matching the pattern
    search_pattern = str(input_dir / '**' / f'wrfout_d{domain}*')
    files = sorted(glob.glob(search_pattern, recursive = True))

    # Check if the list is empty (meaning no files matched the domain pattern)
    if not files:
        logger.error(f'No WRF files found for domain d{domain} in directory: {f_loc}')

    return files

def loc_mask(ds, dataset_type, lat_min = 34.43, lat_max = 46.57, lon_min = -117.74, lon_max = -100.57):
    """
    Mask given dataset based on predefined latitude and longitude values. Dataset type must be WRF, gridMET, or other.
    Default coordinates are for gridMET interpolation (exculdes locations over the Pacific Ocean).
    """

    if dataset_type == 'WRF':
        # Select gridpoints only within the given spatial boundaries
        mask = (
            (ds['XLAT'] >= lat_min) & (ds['XLAT'] <= lat_max) &
            (ds['XLONG'] >= lon_min) & (ds['XLONG'] <= lon_max)
        )

        # Apply mask to ds
        ds_masked = ds.where(mask, drop = True)

    elif dataset_type == 'gridMET':
        # Select gridpoints only within the given spatial boundaries
        ds_masked = ds.sel(
            lat = slice(lat_max, lat_min), 
            lon = slice(lon_min, lon_max)
        )
    
    elif dataset_type == 'other':
        # Select gridpoints only within the given spatial boundaries
        mask = (
            (ds['lat'] >= lat_min) & (ds['lat'] <= lat_max) &
            (ds['lon'] >= lon_min) & (ds['lon'] <= lon_max)
        )

        # Apply mask to ds
        ds_masked = ds.where(mask, drop = True)

    else:
        ds_masked = None
        logger.error('Select valid dataset type: WRF, gridMET, or other')

    logger.debug(f'Output of masked {dataset_type} dataset')
    logger.debug(ds_masked)
    return ds_masked

def interpo_MET(WRF_file, location, var, year):
    """
    Open one year of gridMET data at a time and interpolate to WRF grid.
    """
    # Open test file and pull data
    WRF_ds = xr.open_dataset(WRF_file)

    # Mask WRF data
    WRF_masked = loc_mask(WRF_ds, 'WRF')
    logger.success(f'WRF dataset successfully masked!')

    # Extract lat and long values and squeeze out the time dimension
    lat = WRF_masked['XLAT'].isel(Time = 0).values
    lon = WRF_masked['XLONG'].isel(Time = 0).values

    # Close files to erase from memory
    WRF_ds.close()
    WRF_masked.close()

    # Define file path and open gridMET data for defined variable
    fpath = f'{location}{var}_{year}.nc'
    MET_ds = xr.open_dataset(fpath)[MET_vars[var]] # use MET_vars dictionary to access the variable by the name it's saved to in xarray
    logger.debug(f'gridMET variable in use: {MET_vars[var]}')
    time_vals = pd.date_range(start = f'{year}-01-01', end = f'{year}-12-31', freq = 'D')
    n_times = len(time_vals)

    # Mask gridMET data to size of study region
    MET_masked = loc_mask(MET_ds, 'gridMET')
    logger.success(f'gridMET dataset successfully masked!')

    # Lat values are ordered from max to min and need to be flipped
    MET_masked = MET_masked.isel(lat = slice(None, None, -1))

    # Check that lattitude was successfully reversed
    assert MET_masked.lat.values[0] < MET_masked.lat.values[-1] or logger.error('Latitude slice did not reverse properly.')

    # Keep lat and lon dims as 2d for regridding process
    dims_2d = ('south_north', 'west_east')

    # Create a blueprint for a new dataset to house the interpolated data
    ds_map = xr.Dataset(
        {
            var: (['time', 'lat', 'lon'], MET_masked.values)
        },
        coords = {
            'time': ('time', time_vals),
            'lat': (dims_2d, lat),
            'lon': (dims_2d, lon),
        }
    )

    # Make a regridder to apply to gridMET data
    regridder = xe.Regridder(MET_masked, ds_map, method = 'bilinear', extrap_method = 'inverse_dist')

    # House interpolated DataArray in RAM not on disk
    da_regridded = regridder(MET_masked)

    # Lat and lon dims really should be 3D to match WRF
    dims_3d = ('time', 'south_north', 'west_east')

    # Repeat 2D spatial grid across all time steps
    lat_3d = np.repeat(np.expand_dims(lat, axis = 0), n_times, axis = 0)
    lon_3d = np.repeat(np.expand_dims(lon, axis = 0), n_times, axis = 0)

    # Turn DataArray into Dataset
    ds_out = xr.Dataset(
        {
            var: (['time', 'lat', 'lon'], da_regridded.values)
        },
        coords = {
            'time': ('time', time_vals),
            'lat': (dims_3d, lat_3d),
            'lon': (dims_3d, lon_3d),
        }
    )

    # Close unnecessary files out of memory
    MET_ds.close()
    ds_map.close()
    MET_masked.close()

    logger.success('gridMET successfully interpolated!')
    logger.debug(ds_out[var].values)

    # TODO: make sure we're not losing anything on the edges that takes off the GSL region
    return ds_out

# Make function compatible with with blocks 
@contextmanager
def open_or_skip(fpath):
    """
    Attempt to open a netCDF4 dataset normally and repair if netCDF structure is corrupted and throwing a HDF5 attribute error.
    """ 

    # Pull filename from path
    filename = Path(fpath).name

    try:
        # Attempt to open file normally first
        with xr.open_dataset(fpath) as ds:
            logger.success(f'netCDF file opened successfully: {fpath}')
            
            # YIELD instead of return so the with block can consume it
            yield ds

    except (AttributeError, RuntimeError, OSError) as e:
        logger.warning(f'Error opening file: {fpath}. Data is corrupted.')
        raise FileNotFoundError(f'Corrupted file: {fpath}')
        # If you want you can call repair function here instead

def data_saver(data, destination, var, year):
    """
    Save gridMET data to a netcdf file.
    """
    # Create output directory to store new cleaned files
    output_dir = current_dir / destination / var
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Save to netcdf
    out_path = os.path.join(output_dir, f'{var}_{year}.nc')
    data.to_netcdf(out_path)
    logger.success(f'File saved to: {out_path}')

    # Close out of data once saved
    data.close()
    logger.success(f'{var}-{year} data saved!')

def fix_time_coord(ds):

    # Ensure the time coordinate is always parsed as a proper datetime object
    if 'time' in ds.coords:
        ds['time'] = pd.to_datetime(ds['time'].values)

    return ds

def WRF_daily(today, tomorrow, files, var, domain, current_dir):
    """
    Calculate daily average WRF value for given variable. Daily data is saved as netCDF to output_dir.
    """

    # List of files for given date
    matched_files = [f for f in files if f'wrfout_d{domain}_{today}' in f or f'wrfout_d{domain}_{tomorrow}' in f]

    # Empty list to fill with daily data
    temp_clean = []

    for file in matched_files:
        try:
            # Open one timestamp file at a time
            with open_or_skip(file) as ds:
                
                # Only pull out data for given variable
                var_data = ds[WRF_vars[var]]
                xtime_vals = var_data['XTIME'].values
                lat_vals = ds['XLAT'].values
                lon_vals = ds['XLONG'].values

                # Recognize that dims should be 3D
                dims_3d = ('time', 'south_north', 'west_east')

                # Create new dataset to save data to
                clean_ds = xr.Dataset(
                    {
                        var: (['time', 'lat', 'lon'], var_data.data)
                    },
                    coords = {
                        'time': ('time', xtime_vals),
                        'lat': (dims_3d, lat_vals),
                        'lon': (dims_3d, lon_vals)
                    }
                )
                
                # Save each timestamp worth of day to predefined list
                temp_clean.append(clean_ds)
            
        except FileNotFoundError as e:
            logger.warning(f'Skipping file {file} due to error: {e}')
            continue

    # Concatonate four timestamp files into one file
    combo_clean = xr.concat(temp_clean, dim = 'time')

    # Convert time values from UTC to MT
    localized_time = pd.to_datetime(combo_clean['time'].values).tz_localize(ZoneInfo('UTC')).tz_convert(ZoneInfo('America/Denver'))

    # Strip timestamps of specific timezone data
    clean_format = localized_time.tz_localize(None)
    ds_new_tz = combo_clean.assign_coords(time = ('time', clean_format))
    logger.info(ds_new_tz['time'])

    # Filter dates to match today
    target_date = datetime.strptime(today, '%Y-%m-%d').date()
    logger.info(f'Filtering for {target_date}')
    mask = (clean_format.date == target_date)
    time_clean_ds = ds_new_tz.isel(time = mask)
    logger.info(time_clean_ds['time'])

    # Check that only four timestamps are included in the daily data
    if len(time_clean_ds['time']) != 4:
        logger.error(f'{today} only has {len(time_clean_ds["time"])} timestamps instead of 4. Some files might be corrupted or missing.')

    if var == 'tmmx':
        # Select daily max along XTIME dim for every gridpoint
        adj_data = time_clean_ds[var].max(dim = 'time')

        # Expand time dim back out after it was collapsed
        daily_data = adj_data.expand_dims('time')

    elif var == 'tmmn':
        # Select daily min along XTIME dim for every gridpoint
        adj_data = time_clean_ds[var].min(dim = 'time')
        # TODO: check if this is most efficient way because it seems to be slower

        # Expand time dim back out after it was collapsed
        daily_data = adj_data.expand_dims('time')

    elif var == 'pr':
        # WRF stores precipitation as a continuously increasing staircase - difference first and last stairstep in a day to get true precip value for that day
        adj_data= time_clean_ds.isel(time = -1) - time_clean_ds.isel(time = 0) # -1 grabs last index regardless of how many timestamps there are
        daily_data = adj_data[var].expand_dims('time')

        # Verify that precipitation values only are positive
        assert (daily_data >= 0).all() or logger.error('Found negative precipitation values.')

    elif var == 'sph':
        # Find daily average 
        daily_avg = time_clean_ds[var].mean(dim = 'time')
        
        # Convert Q2 to specific humidity
        adj_data = daily_avg / (1 + daily_avg)

        # Expand time dim back out after it was collapsed
        daily_data = adj_data.expand_dims('time')

    else:
        # Calculate daily average
        adj_data = time_clean_ds[var].mean(dim = 'time')

        # Expand time dim back out after it was collapsed
        daily_data = adj_data.expand_dims('time')
    
    # Check that daily_data is xr.DataArray not xr.Dataset
    assert isinstance(daily_data, xr.DataArray) or logger.error('Type must be xr.DataArray')

    # Convert date to a pandas datetime object
    time_dt = pd.to_datetime(today)

    # Keep lat and lon dims as 2d for regridding process
    dims_2d = ('south_north', 'west_east')
    
    # Save daily data to a new dataset
    daily_ds = xr.Dataset(
        {
            var: (['time', 'south_north', 'west_east'], daily_data.values)
        },
        coords = {
            'time': ('time', [time_dt]),
            'lat': (dims_2d, lat_vals),
            'lon': (dims_2d, lon_vals)
        }
    )

    # Assign global attributes to new dataset
    daily_ds.attrs = ds.attrs.copy()

    # Assign variable specific attributes to new dataset
    daily_ds[var].attrs = ds[WRF_vars[var]].attrs.copy()

    logger.debug(daily_ds)

    # Create output directory to store new cleaned files
    output_dir = current_dir / 'daily' / var 
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Save to netcdf
    out_path = os.path.join(output_dir, f'daily_{var}_{today}.nc')
    daily_ds.to_netcdf(out_path)
    logger.success(f'File saved to: {out_path}')

    # Close daily files out of memory
    daily_ds.close()

    return daily_ds

# Catch silent errors and report to log file
@logger.catch 
def main(variable, domain, WRF_in, MET_in):
    # State what variable is being used
    logger.info(f'Debiasing for {variable}.')

    if variable == 'wind':
        variables = ['uas', 'vas']
    else:
        variables = [variable]

    for var in variables:
        # TODO: log errors if the workflow isn't completed sequentially
        # Generate list of WRF input files
        files = get_fpaths(WRF_in, domain)
        hist_period = range(1985, 2015) # Set to historical period for obs data
        
        # Set to full historical period
        for year in hist_period:
            # Call and interpolate gridMET data (obs) for the given year 
            MET_data = interpo_MET(files[0], MET_in, var, year) # pass first WRF file in files as example grid

            # Save year of interpolated gridMET data
            save_data = data_saver(MET_data, 'gridMET', var, year)

        # Combine all gridMET files into a single file with context manager
        with xr.open_mfdataset(
            glob.glob(str(current_dir / 'gridMET' / var / '*.nc')), 
            combine = 'nested', 
            concat_dim = 'time', 
            preprocess = fix_time_coord
            ) as gridmet_open:

            # Write file to netcdf
            gridmet_open.sortby('time').to_netcdf(current_dir / 'gridMET' / f'gridMET_GSLBIP_{var}.nc')
        
        logger.success(f'All gridMET files successfully interpolated and saved for {var}!')

        # Set to historical + future period
        model_period = range(1985, 2100)
        for year in model_period:
            # Create date range using pandas
            # TODO: set to dates for full year
            dates = pd.date_range(start = f'{year}-01-01', end = f'{year}-12-31', freq = 'D') 

            for day in dates:
                if day == dates[0]:
                    # Set the day you want to be working with to today
                    today = day.strftime('%Y-%m-%d') # Turn day in to usable date string 
                    continue

                # Handle the last day of the year
                elif day == dates[-1]:
                    # Because of the offset from UTC to MT you need to pull in the next day worth of data as well
                    tomorrow = day.strftime('%Y-%m-%d')

                    # Create clean file of daily WRF data
                    daily_avg = WRF_daily(today, tomorrow, files, var, domain, current_dir)

                    # Set tomorrow as the new today to move on to the next series
                    today = tomorrow

                    # Check that you aren't calling data outside of the model's date range
                    if (year + 1) not in model_period:
                        tomorrow = f'{year}-12-31'
                    else: 
                        tomorrow = f'{year + 1}-01-01'

                    # Create clean file of daily WRF data for the last day in the date range
                    logger.info(f'Today for 12-31 is {today} and tomorrow is {tomorrow}.')
                    daily_avg = WRF_daily(today, tomorrow, files, var, domain, current_dir)

                else:
                    # Because of the offset from UTC to MT you need to pull in the next day worth of data as well
                    tomorrow = day.strftime('%Y-%m-%d')

                    # Create clean file of daily WRF data
                    daily_avg = WRF_daily(today, tomorrow, files, var, domain, current_dir)

                    # Set tomorrow as the new today to move on to the next series
                    today = tomorrow

# ======================
# ---- Entry Point ----
# ======================

if __name__ == '__main__':
    # Track program time in log files
    start = time.perf_counter()
    logger.info(f'Beginning execution of data preprocessing for {os.environ.get("variable")}.')

    # TODO: check that all 12-31's got included - gridMET seems to not have this issue just WRF
    # Call inputs from slurm script
    main(
        variable = os.environ.get("variable"),
        domain = os.environ.get("domain"),
        WRF_in = os.environ.get("WRF_in"), 
        MET_in = os.environ.get("MET_in")
        )

    # Report of runtime at completion 
    logger.success(f'Debiasing process completed!')
    elapsed = time.perf_counter() - start

    if elapsed < 60: 
        logger.info(f'Total runtime: {elapsed:.4f} sec')
    elif elapsed < 3600:
        logger.info(f'Total runtime: {elapsed / 60:.2f} min')
    else:
        logger.info(f'Total runtime: {elapsed / 3600:.2f} hr')

    # Force script to stop running once code is finished
    sys.exit(0)
