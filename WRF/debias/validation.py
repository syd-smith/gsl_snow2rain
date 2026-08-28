# %%
"""
Author: Sydney Smith
Date Created: August 25, 2026
"""

import datetime as dt
import glob
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import xarray as xr
from zoneinfo import ZoneInfo

# %%

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from old.temporal_chunks import open_or_skip, get_fpaths


dates = pd.date_range(start = '1985-01-01', end = '1985-12-31', freq = 'D') 
WRF_in = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/'
today = dates[0].strftime('%Y-%m-%d')
tomorrow = dates[1].strftime('%Y-%m-%d')
domain = '03'
var = 'tmmn'
files = get_fpaths(WRF_in, domain)

WRF_vars = {
    'tmmn': 'T2',
    'tmmx': 'T2',
    'pr': 'RAINNC',
    'sph': 'Q2',
    'srad': 'SWDOWN',
    'vas': 'V10',
    'uas': 'U10',
}

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
        # logger.warning(f'Skipping file {file} due to error: {e}')
        continue

# Concatonate four timestamp files into one file
combo_clean = xr.concat(temp_clean, dim = 'time')

# %%


localized_time = pd.to_datetime(combo_clean['time'].values).tz_localize(ZoneInfo('UTC')).tz_convert(ZoneInfo('America/Denver'))
clean_format = localized_time.tz_localize(None)
combo_clean = combo_clean.assign_coords(time = ('time', clean_format))
target_date = dt.datetime.strptime(today, '%Y-%m-%d').date()
mask = localized_time.date == target_date
ds_time_sel = combo_clean.isel(time = mask)

# %%
if var == 'tmmx':
    # Select daily max along XTIME dim for every gridpoint
    adj_data = combo_clean[var].max(dim = 'time')

    # Expand time dim back out after it was collapsed
    daily_data = adj_data.expand_dims('time')

elif var == 'tmmn':
    # Select daily min along XTIME dim for every gridpoint
    adj_data = combo_clean[var].min(dim = 'time')
    # TODO: check if this is most efficient way because it seems to be slower

    # Expand time dim back out after it was collapsed
    daily_data = adj_data.expand_dims('time')

elif var == 'pr':
    # WRF stores precipitation as a continuously increasing staircase - difference first and last stairstep in a day to get true precip value for that day
    adj_data= combo_clean.isel(time = -1) - combo_clean.isel(time = 0) # -1 grabs last index regardless of how many timestamps there are
    daily_data = adj_data[var].expand_dims('time')

elif var == 'sph':
    # Find daily average 
    daily_avg = combo_clean[var].mean(dim = 'time')
    
    # Convert Q2 to specific humidity
    adj_data = daily_avg / (1 + daily_avg)

    # Expand time dim back out after it was collapsed
    daily_data = adj_data.expand_dims('time')

else:
    # Calculate daily average
    adj_data = combo_clean[var].mean(dim = 'time')

    # Expand time dim back out after it was collapsed
    daily_data = adj_data.expand_dims('time')

time_dt = pd.to_datetime(target_date)

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





# %%

# def trend_plt():

# def cdf_plt():
    # obs, raw hist, raw fut, debiased hist and fut (5 cdfs total)

def avg_annual_scatter(data, var, fig, ax):
    """
    Create a scatter plot showing the annual cycle of given data over the entire spatial region.
    """

    # Average data to get 365 x 1 x 1 (day of year x lat x lon)
    dayOyear = data[var].groupby('time.dayofyear').mean('time')
    spatial_avg = dayOyear.mean(dim = ['lat', 'lon'])

    # Plot scatter data
    ax.plot(
        spatial_avg['time'],
        spatial_avg[var].values, 
        'ko',
        alpha = 0.4
    )

    # Adjust plot format settings
    ax.set_title(f'Mean Annual Cycle of {var} Across Study Region')
    ax.set_ylabel(var)
    ax.set_xlabel('Time')
    ax.grid(True, linestyle = '--', alpha = 0.5)
    ax.legend(frameon = True)

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
        return data

        


# def elevation_scatter():
    

def main(var, wrf_output_location):

    test = elevation_data(wrf_output_location)
    print(test)
    # # Open debiased data
    # debiased = xr.open_dataset

    # # Initialize plot
    # fig, ax = plt.subplots(figsize = (12, 6))



# ======================
# ---- Entry Point ----
# ======================

if __name__ == '__main__':
    main(
        var = 'tmmn', 
        wrf_output_location = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014'
    )

