"""
Author: Sydney Smith & Austin LaMontagne
Date Created: July 24, 2026
"""

from contextlib import contextmanager
from datetime import datetime
import glob
from loguru import logger
import netCDF4
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import scipy
import sys
import time
import xarray as xr
import xesmf as xe
from zoneinfo import ZoneInfo


# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
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
    'pr': 'RAINNC',
    'sph': 'Q2',
    'srad': 'rsds',
    'vas': 'V10',
    'uas': 'U10',
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
    dims_2d = ('south_north', 'east_west')

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
    dims_3d = ('time', 'south_north', 'east_west')

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

def file_repair(fpath):
    """
    Repair corrupted netcdf. Won't work on files if Time dim is zero.
    """
    # Pull filename from path
    filename = Path(fpath).name

    # Attempt repair on netCDF file using h5py
    try:
        # Dynamically find a valid template file in the same directory
        dir_name = Path(fpath).parent
        available_files = sorted(list(dir_name.glob('wrfout_d03_*')))
        
        template_path = None
        for candidate in available_files:
            if candidate.name != filename: # avoid using the broken file itself
                template_path = candidate
                logger.info(f'Using template file for repair: {template_path}')
                break
        
        # Raise error if template is not found
        if not template_path or not template_path.exists():
            error_msg = f'No valid template file found in {dir_name}'
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        with xr.open_dataset(template_path) as ds_template:
            ds_fixed = ds_template.load()
            logger.success(f'Loaded template dataset from {template_path} for structure reference.')
                
        # Open the corrupted file via netCDF4 to extract its raw data variables
        with netCDF4.Dataset(fpath, 'r') as nc:
            for var_name in list(ds_fixed.data_vars) + list(ds_fixed.coords):
                if var_name in ['Times', 'XTIME', 'DateStrLen']: # Will be handled below
                    continue
                    
                if var_name in nc.variables:
                    raw_data = nc.variables[var_name][:]
                    
                    # Manually fix Time if it has zero length
                    if 'Time' in ds_fixed[var_name].dims and raw_data.size > 0:
                            time_axis = ds_fixed[var_name].get_axis_num('Time')
                            if time_axis < raw_data.ndim and raw_data.shape[time_axis] == 0:
                                new_shape = list(raw_data.shape)
                                new_shape[time_axis] = 1
                                raw_data = np.zeros(new_shape, dtype=raw_data.dtype)
                            
                    ds_fixed[var_name].values = raw_data

        # Pull timestamp information from filename to update corrupted Times and XTIME
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', filename)
        if match:
            time_str = match.group(1)

            # Update Times to match: array([b'1985-02-06_00:00:00'], dtype='|S19')
            if 'Times' in ds_fixed:
                ds_fixed['Times'].values = np.array([time_str.encode('utf-8')], dtype='|S19')
                
            # Update XTIME to match: array(['1985-02-06T00:00:00.000000000'], dtype='datetime64[ns]')
            if 'XTIME' in ds_fixed:
                iso_time_str = time_str.replace('_', 'T')
                ds_fixed['XTIME'].values = np.array([iso_time_str], dtype='datetime64[ns]')
        
        # CRITICAL FIX: Ensure native WRF variables are coords, not data vars
        wrf_coords = ['XTIME', 'XLAT', 'XLONG']
        for coord_name in wrf_coords:
            if coord_name in ds_fixed.variables and coord_name not in ds_fixed.coords:
                ds_fixed = ds_fixed.set_coords(coord_name)

        # Yield ds so it can be used in with block
        ds_fixed = ds_fixed.load()  # Load into memory to avoid lazy loading issues
        logger.success(f'{filename} repaired successfully.')

        print(ds_fixed)
        print(ds_fixed[WRF_vars[var]])
        print(ds_fixed[WRF_vars[var]].XLAT.values)
        yield ds_fixed

    except Exception as e:
        logger.exception(f'File {filename} might be completely corrupted at the HDF5 layer: {e}')
        # TODO: turn continue on once bugs are fixed
        # logger.warning(f'Skipping corrupted file {file}: {e}')
        # continue

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

                # Recognize that dims should be 2D
                dims_3d = ('time', 'south_north', 'east_west')

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

    # Filter dates to match today
    target_date = datetime.strptime(today, '%Y-%m-%d').date()
    mask = localized_time.date == target_date
    time_clean_ds = combo_clean.isel(time = mask)

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
    
    # Save daily data to a new dataset
    daily_ds = xr.Dataset(
        {
            var: (['time', 'lat', 'lon'], daily_data.values)
        },
        coords = {
            'time': ('time', [time_dt]),
            'lat': (dims_3d, lat_vals),
            'lon': (dims_3d, lon_vals)
        }
    )

    # Assign global attributes to new dataset
    daily_ds.attrs = ds.attrs.copy()

    # Assign variable specific attributes to new dataset
    daily_ds[var].attrs = ds[WRF_vars[var]].attrs.copy()

    logger.debug(daily_ds)

    # Create output directory to store new cleaned files
    output_dir = current_dir / 'wrfout' / var 
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Save to netcdf
    out_path = os.path.join(output_dir, f'daily_{var}_{today}.nc')
    daily_ds.to_netcdf(out_path)
    logger.success(f'File saved to: {out_path}')

    # Close daily files out of memory
    daily_ds.close()

    return daily_ds

def bias_daily(WRF_data, MET_data, date, var):
    """
    Calculate the daily bias (WRF - gridMET) and save data to netCDF.
    """
    # Calculate bias
    if var == 'pr':
        daily_bias = (WRF_data + 0.1) / (MET_data + 0.1)

        #artificially build out pr ratios
        #if both values below 0.1 the set ratio to 1 (no change)
        # add small offset for gridmet so you never get a divide by zero error?
        
    else:
        daily_bias = WRF_data - MET_data

    # Create output directory to store new cleaned files
    output_dir = current_dir / 'bias' / var
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Save to netcdf
    out_path = os.path.join(output_dir, f'bias_{var}_{date}.nc')
    daily_bias.to_netcdf(out_path)
    logger.success(f'File saved to: {out_path}')

    return daily_bias

def climate_avg(var):
    """
    Calculate the climatological average (across historical period) bias for a given day.
    """
    # Location of bias data from bias_daily
    bias_data = current_dir / 'bias' / var

    # Create output directory to store new cleaned files
    output_dir = current_dir / 'climate_avg' / var
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # TODO: set to correct date range (full year)
    # Generate normal date range with a dummy year (note 1985 is not a leap year)
    dates = pd.date_range(start = '1985-01-01', end = '1985-12-31', freq = 'D')

    # Strip out the year and keep only month-day
    month_days = dates.strftime('%m-%d')

    for day in month_days:
        list_of_files = sorted(glob.glob(f'{bias_data}/bias_{var}_*{day}.nc'))
        # TODO: create error log if list is empty
        all_years = xr.open_mfdataset(list_of_files, combine = 'nested', concat_dim = 'time')

        # Take the mean of all years in the historical period
        climate_avg_of_day = all_years.mean(dim = 'time').load() # computes mean into RAM and detaches it from files
        logger.debug(climate_avg_of_day)

        # Mannual add back time dimension (1985 is just a placeholder year)
        date_add_back = climate_avg_of_day.assign_coords(time = pd.to_datetime(f'1985-{day}'))
        logger.debug(date_add_back['time'].values)

        # Reconstruct proper lat and lon dimensions
        lat_2d = all_years['lat'].isel(time = 0)
        lon_2d = all_years['lon'].isel(time = 0)
        date_add_back['lat'] = lat_2d
        date_add_back['lon'] = lon_2d   

        # Save to netcdf
        out_path = os.path.join(output_dir, f'climate_avg_{var}_{day}.nc')
        date_add_back.to_netcdf(out_path)
        logger.success(f'File saved to: {out_path}')

        # Close files out of memory
        all_years.close()
        climate_avg_of_day.close()
        date_add_back.close()

# Catch silent errors and report to log file
@logger.catch 
def main(var, domain, WRF_in, MET_in):
    # State what variable is being used
    logger.info(f'Debiasing for {var}.')
    # TODO: Build in today and tomorrow time handling (see spatial_chunks.py -> main())

    # # TODO: log errors if the workflow isn't completed sequentially
    # # Generate list of WRF input files
    # files = get_fpaths(WRF_in, domain)
    
    # # TODO: set to full historical period
    # for year in range(1985, 2015):
    #     # Call and interpolate gridMET data for the given year 
    #     MET_data = interpo_MET(files[0], MET_in, var, year) # pass first file in files as example grid

    #     # Create date range using pandas
    #     # TODO: set to dates for full year
    #     dates = pd.date_range(start = f'{year}-01-01', end = f'{year}-12-31', freq = 'D') 

    #     for day in dates:
    #         # Turn day in to usable date string 
    #         day_str = day.strftime('%Y-%m-%d')

    #         # Create clean file of daily WRF data
    #         daily_avg = WRF_daily(day_str, files, var, domain, current_dir)

    #         # Select day worth of gridMET data
    #         MET_select = MET_data.sel(time = day_str)

    #         # Find bias for given data
    #         bias = bias_daily(daily_avg, MET_select, day_str, var)

    #         # Close daily files out of memory
    #         bias.close()

    #     # Close out of gridMET data once the entire year is complete
    #     MET_data.close()
    #     logger.success(f'{year} daily bias caclulations complete!')
    
    # Calculate the climatological daily bias
    # climate_avg(var)
    # logger.success('Climatological averages complete!')

    # # Calculate fourier coefficients using custom harmonic function (multiple linear regression)
    # # Outputs saved the harmonics directory
    # for order in range(1, 4):
    #     apply_harmonic(var, order)

    # Pass coefficients back into custom harmonic function to get smoothed data


# ======================
# ---- Entry Point ----
# ======================

# Set variable based on gridMET variable save names
# tmmn, tmmx, pr, sph, srad, vas, uas

if __name__ == '__main__':
    # Track program time in log files
    start = time.perf_counter()
    logger.info('Beginning execution.')

    # Only inputs required
    main(
        var = 'uas',
        domain = '03',
        WRF_in = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/', 
        MET_in = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/'
        )

    # Report of runtime at completion 
    logger.info(f'Total runtime: {time.perf_counter() - start:.4f}s')

