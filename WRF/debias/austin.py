"""
Author: Sydney Smith & Austin LaMontagne
Date Created: July 24, 2026
"""

import glob
import pandas as pd
import xarray as xr
import xesmf as xe

# Dictionary that calls name of WRF variable based on the given gridMET variable
WRF_vars = {
    "tmmn": "T2",
    "tmmx": "T2",
    "pr": "RAINNC",
    "sph": "Q2",
    "srad": "SWDOWN",
    "vas": "V10",
    "uas": "U10",
}

# Dictionary that calls name of variable that gridMET data is saved to within the xarray Dataset based on the variable name the file is saved to
MET_vars = {
    "tmmn": "air_temperature",
    "tmmx": "air_temperature",
    "pr": "precipitation_amount",
    "sph": "specific_humidity",
    "srad": "surface_downwelling_shortwave_flux_in_air",
    "vas": "vas",
    "uas": "uas",
}


def get_fpaths(f_loc):
    """
    Input path to directory that contains WRF data. Outputs a sorted list of file names.
    """
    # Create list to store file names in
    files = sorted(glob.glob(f'{f_loc}/wrfout_d03*'))

    return files

def loc_mask(ds, dataset_type):
    """
    Mask given Dataset based on predefined latitude and longitude values. Dataset type must be WRF or gridMET.
    """
    # Define boundaries for interpolation (exculde locations over the Pacific Ocean)
    lat_min, lat_max = 34.43, 46.57
    lon_min, lon_max = -117.74, -100.57

    if dataset_type == 'WRF':
        # Select gridpoints only within the given spatial boundaries
        mask = (
            (ds['XLAT'] >= lat_min) & (ds['XLAT'] <= lat_max) &
            (ds['XLONG'] >= lon_min) & (ds['XLONG'] <= lon_max)
        )

        # Apply mask to ds
        ds_masked = ds.where(mask, drop=True)

    elif dataset_type == 'gridMET':
        # Select gridpoints only within the given spatial boundaries
        ds_masked = ds.sel(
            lat = slice(lat_max, lat_min), 
            lon = slice(lon_min, lon_max)
        )

    else:
        # todo: log error here
        print('Select valid dataset type.')
        ds_masked = None

    return ds_masked

def interpo_MET(WRF_file, location, var, year):
    """
    Open one year of gridMET data at a time and interpolate to WRF grid.
    """
    # Open test file and pull data
    WRF_ds = xr.open_dataset(WRF_file)

    # Mask WRF data
    WRF_masked = loc_mask(WRF_ds, 'WRF')

    # Extract lat and long values and squeeze out the time dimension
    lat = WRF_masked['XLAT'].isel(Time = 0).values
    lon = WRF_masked['XLONG'].isel(Time = 0).values

    # Close files to erase from memory
    WRF_ds.close()
    WRF_masked.close()

    # Define file path and open gridMET data for defined variable
    fpath = f'{location}{var}_{year}.nc'
    MET_ds = xr.open_dataset(fpath)[MET_vars[var]] # use MET_vars dictionary to access the variable by the name it's saved to in xarray
    time_vals = MET_ds['day']

    # Mask gridMET data
    MET_masked = loc_mask(MET_ds, 'gridMET')

    # Lat values are ordered from max to min and need to be flipped
    MET_masked = MET_masked.isel(lat = slice(None, None, -1))

    # Recognize that dims should be 3D
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

    # Turn DataArray into Dataset
    ds_out = xr.Dataset(
        {
            var: da_regridded.values
        },
        coords = {
            'time': ('time', time_vals),
            'lat': (dims_2d, lat),
            'lon': (dims_2d, lon),
        }
    )

    # Close unnecessary files out of memory
    MET_ds.close()
    ds_map.close()
    MET_masked.close()

    return ds_out

def WRF_daily(date, files, var):
    """
    Calculate daily average WRF value for given variable.
    """

    # List of files for given date
    matched_files = [f for f in files if f'wrfout_d03_{date}' in f]

    # Create output directory to store new cleaned files
    output_dir = f'/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/daily/{var}/'
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Empty list to fill with daily data
    temp_clean = []

    for file in matched_files:
        # Open one timestamp file at a time
        with xr.open_dataset(file) as ds:
            # Only pull out 2 meter temperature data
            var_data = ds[WRF_vars[var]]
            xtime_vals = ds['XTIME'].values
            lat_vals = ds['XLAT'].isel(XTIME = 0).values
            lon_vals = ds['XLONG'].isel(XTIME = 0).values

            # Recognize that dims should be 2D
            dims_2d = ('south_north', 'east_west')

            # Create new dataset to save data to
            clean_ds = xr.Dataset(
                {
                    var: (['time', 'lat', 'lon'], var_data.data)
                },
                coords = {
                    'time': ('time', xtime_vals),
                    'lat': (dims_2d, lat_vals),
                    'lon': (dims_2d, lon_vals)
                }
            )
            
            # Save each timestamp worth of day to predefined list
            temp_clean.append(clean_ds)

    # Concatonate four timestamp files into one file
    combo_clean = xr.concat(temp_clean, dim = 'time')
    
    if var == 'tmmx':
        # Select daily max along XTIME dim for every gridpoint
        adj_data = combo_clean[var].max(dim = 'time')

    elif var == 'tmmn':
        # Select daily min along XTIME dim for every gridpoint
        adj_data = combo_clean[var].min(dim = 'time')

    elif var == 'pr':
        # WRF stores precipitation as a continuously increasing staircase - difference first and last stairstep in a day to get true precip value for that day
        adj_data = combo_clean_ds[var].sel(time = 0) - combo_clean[var].sel(XTIME = 3)

    elif var == 'sph':
        # Find daily average 
        daily_avg = combo_cleam[var].mean(dim = 'time')
        
        # Convert Q2 to specific humidity
        adj_data = daily_avg / (1 + daily_avg)

    else:
        # Calculate daily average
        daily_adj = combo_clean[var].mean(dim = 'time')
    
    # Expand time dim back out after it was collapsed
    daily_data = adj_data.expand_dims('time')

    # Save daily data to a new dataset
    daily_ds = xr.Dataset(
        {
            var: (['time', 'lat', 'lon'], daily_data.data)
        },
        coords = {
            'time': ('time', xtime_vals),
            'lat': (dims_2d, lat_vals),
            'lon': (dims_2d, lon_vals)
        }
    )

    # Save to netcdf
    out_path = os.path.join(output_dir, f'{var}_{date}.nc')
    daily_ds.to_netcdf(out_path)

    return daily_ds






def main():
    # Set variable based on gridMET variable save names
    # tmmn, tmmx, pr, sph, srad, vas, uas
    var = 'pr'

    # Locations for input data
    WRF_in = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/'
    MET_in = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/'

    # List of WRF input files
    files = get_fpaths(WRF_in)
    
    # todo: set to full historical period
    for year in range(1985, 1986):
        # Call and interpolate gridMET data for the given year 
        MET_data = interpo_MET(files[0], MET_in, var, year)
        print(MET_data)

        # # Create date range using pandas
        # # todo: set to dates for full year
        # dates = pd.date_range(start = f'{year}-01-01', end = f'{year}-01-02', freq = 'D') 

        # for day in dates:
        #     # Turn day in to usable date string 
        #     day_str = day.strftime('%Y-%m-%d')

        #     # Create files with daily average WRF data
        #     test = WRF_daily(day_str, files, var)

        #     MET_select = MET_data.sel(time = day_str)
        #     print(MET_select)


        # MET_data.close()


        

    



if __name__ == '__main__':
    main()

