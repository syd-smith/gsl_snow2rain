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

    # Mask gridMET data
    MET_masked = loc_mask(MET_ds, 'gridMET')

    # Lat values are ordered from max to min and need to be flipped
    MET_masked = MET_masked.isel(lat = slice(None, None, -1))

    # Create a blueprint for a new dataset to house the interpolated data
    ds_map = xr.Dataset(
        {
            var: (['time', 'lat', 'lon'], MET_masked.values)
        },
        coords = {
            'time': (MET_masked['day']),
            'lat': (('south_north', 'west_east'), lat),
            'lon': (('south_north', 'west_east'), lon),
        }
    )

    # Make a regridder to apply to gridMET data
    regridder = xe.Regridder(MET_masked, ds_map, method = 'bilinear', extrap_method = 'inverse_dist')

    # House interpolated data in RAM not on disk
    ds_out = regridder(MET_masked)

    # Close unnecessary files out of memory
    MET_ds.close()
    ds_map.close()
    MET_masked.close()

    return ds_out

def WRF_daily(MET_data, f_list, var):
    """
    Calculate daily average WRF value for given variable.
    """

    # Grab one day worth of output files
    for start_idx in range(0, len(f_list), 4):
        # Create group out of all of the files for that day
        group = f_list[start_idx : start_idx + 4]

        # Create list to temporarilty store clean files in
        temp_clean = []

        # Loop through every timestamp in the given day
        for time_slice in group:
            # Open timestamp file
            with xr.open_dataset(time_slice) as ds:
                # Only pull out 2 meter temperature data
                var_da = ds[WRF_vars[var]]
                xtime_vals = ds['XTIME'].values
                lat_vals = ds['XLAT'].values
                lon_vals = ds['XLONG'].values

                # Recognize that dims should be 3D
                dims_3d = ('XTIME', 'south_north', 'east_west')

                # Create new dataset to save data to
                clean_ds = xr.Dataset(
                    {
                        var: (dims_3d, var_da.data)
                    },
                    coords = {
                        'XTIME': ('XTIME', xtime_vals),
                        'XLAT': (dims_3d, lat_vals),
                        'XLONG': (dims_3d, lon_vals)
                    }
                )
                
                # Save each timestamp worth of day to predefined list
                temp_clean.append(clean_ds)

        # Concatonate four timestamp files into one file
        combo_clean = xr.concat(temp_clean, dim = 'XTIME')
        
        if var == 'tmmx':
            # Select daily max along XTIME dim for every gridpoint
            daily_data = combo_clean[var].max(dim = 'XTIME')

            # Expand time dim back out after it was collapsed
            daily_data = ds_max.expand_dims('XTIME')

        elif var == 'tmmn':
            # Select daily min along XTIME dim for every gridpoint
            daily_data = combo_clean[var].min(dim = 'XTIME')

            # Expand time dims back out after they were collapsed
            daily_data = ds_min.expand_dims('XTIME')

        # Save daily data to a new dataset
        daily_ds = xr.Dataset(
            {
                var: (dims_3d, daily_data.data)
            },
            coords = {
                'XTIME': ('XTIME', xtime_vals),
                'XLAT': (dims_3d, lat_vals),
                'XLONG': (dims_3d, lon_vals)
            }
        )

        # Save to netcdf
        output = os.path.join(output_dir_min, f'clean_{var}_{start_idx:07d}.nc')
        daily_ds.to_netcdf(output)




def main():

    WRF_in = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/'
    MET_in = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/'
    # Run T2 for tmmn and tmmx separately
    var = 'vas'

    # NEED TO KNOW
    # 1. input file locations
    # 2. variables in WRF
    # 3. variables in gridMET
    
    files = get_fpaths(WRF_in)
    
    for year in range(1985, 1986):
        # # Call and interpolate gridMET data for the given year 
        # MET_data = interpo_MET(files[0], MET_in, var, year)

        # Create date range using pandas
        # todo: ensure the proper dates are inputted
        dates = pd.date_range(start = f'{year}-01-01', end = f'{year}-01-05', freq = 'D') 

        for day in dates:

            day_str = day.strftime('%Y-%m-%d')
            matched_files = [f for f in files if f'wrfout_d03_{day_str}' in f]
            print(matched_files)


        

    



if __name__ == '__main__':
    main()

