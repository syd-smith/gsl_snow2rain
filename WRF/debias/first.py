# %%
"""
Author: Sydney Smith
Date Created: July 23, 2026
"""

import argparse
import os
import xarray as xr

# Change to debias parent directory
os.chdir('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias')

# Pull varibles from shell script
parser = argparse.ArgumentParser()
parser.add_argument('--dir_corrected', required = True)
parser.add_argument('--vname', required = True)
parser.add_argument('--vnew', required = True)

# Set args to access varibales cleanly by name
args = parser.parse_args()

# Create list to store file names in
f_list = []

# read list of files and make into readable string for xarray
with open('file_list.txt', 'r') as file:
    for line in file.readlines():
        f_path = line.strip()
        f_list.append(f_path) # Store file path to list 

# Create list for checking how many files were saved
clean_file_paths = []

if args.vname == 'T2':
    # Create directories to store new cleaned files
    output_dir_max = f'{args.dir_corrected}/tmmx/'
    os.makedirs(output_dir_max, exist_ok = True) # Don't make if it already exists
    output_dir_min = f'{args.dir_corrected}/tmmn/'
    os.makedirs(output_dir_min, exist_ok = True)

    # Grab one day worth of output files
    for idx in range(0, len(f_list), 4):
        # Create group out of all of the files for that day
        group = f_list[idx : idx + 4]
        
        # Create list to temporarilty store clean files in
        temp_clean = []

        # Loop through every timestamp in the given day
        for time_slice in group:
            # Open timestamp file
            with xr.open_dataset(time_slice) as ds:
                # Only pull out 2 meter temperature data
                var_da = ds[args.vname]
                xtime_vals = ds['XTIME'].values
                lat_vals = ds['XLAT'].values
                lon_vals = ds['XLONG'].values

                # Recognize that dims should be 3D
                dims_3d = ('XTIME', 'south_north', 'east_west')

                # Create new dataset to save data to
                clean_ds = xr.Dataset(
                    {
                        args.vname: (dims_3d, var_da.data)
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

        # Select daily max along XTIME dim for every gridpoint
        ds_max = combo_clean['T2'].max(dim = 'XTIME')

        # Expand time dim back out after it was collapsed
        ds_max_expanded = ds_max.expand_dims('XTIME')

        # Create new dataset
        clean_max = xr.Dataset(
            {
                'tmmx': (dims_3d, ds_max_expanded.data)
            },
            coords = {
                'XTIME': ('XTIME', xtime_vals),
                'XLAT': (dims_3d, lat_vals),
                'XLONG': (dims_3d, lon_vals)
            }
        )

        # Save to netcdf
        out_max = os.path.join(output_dir_max, f'clean_tmmx_{idx:07d}.nc')
        clean_max.to_netcdf(out_max)
        clean_file_paths.append(out_max)

        # Select daily min along XTIME dim for every gridpoint
        ds_min = combo_clean['T2'].min(dim = 'XTIME')

        # Expand time dims back out after they were collapsed
        ds_min_expanded = ds_min.expand_dims('XTIME')

        clean_min = xr.Dataset(
            {
                'tmmn': (dims_3d, ds_min_expanded.data)
            },
            coords = {
                'XTIME': ('XTIME', xtime_vals),
                'XLAT': (dims_3d, lat_vals),
                'XLONG': (dims_3d, lon_vals)
            }
        )

        # Save to netcdf
        out_min = os.path.join(output_dir_min, f'clean_tmmn_{idx:07d}.nc')
        clean_min.to_netcdf(out_min)
        clean_file_paths.append(out_min)

else:

    # Create a directory to store the new cleaned files
    output_dir = f'{args.dir_corrected}{args.vnew}/'
    os.makedirs(output_dir, exist_ok = True) # Don't create directory if it already exists

    for i, filepath in enumerate(f_list):
        # Use a context manager so memory is freed immediately after each file closes
        with xr.open_dataset(filepath) as ds:
            
            # Isolate only variable of interest and it's necessary dimensions from the massive dataset to save memory
            var_da = ds[args.vname]
            xtime_vals = ds['XTIME'].values
            lat_vals = ds['XLAT'].values
            lon_vals = ds['XLONG'].values

        # Define the exact dimensions for these arrays
            dims_3d = ('XTIME', 'south_north', 'west_east')
            
            # Build the clean dataset with explicit dimension tuples for every variable/coord
            clean_ds = xr.Dataset(
                {
                    args.vnew: (dims_3d, var_da.data)
                },
                coords = {
                    'XTIME': ('XTIME', xtime_vals),
                    'XLAT': (dims_3d, lat_vals),
                    'XLONG': (dims_3d, lon_vals)
                }
            )
            
            # Define file save path
            out_path = os.path.join(output_dir, f'clean_{args.vnew}_{i:07d}.nc')

            if args.vname == 'RAINNC' and i == 0:
                # Zero out the first file of data
                zero_data = xr.zeros_like(clean_ds[args.vnew])
                zero_ds = clean_ds.copy()
                zero_ds[args.vnew] = zero_data

                # Save the zeroed dataset to netcdf
                zero_ds.to_netcdf(out_path)
                clean_file_paths.append(out_path)

                # Store data for next iteration
                prev_ds = zero_ds

            elif args.vname == 'RAINNC' and i >= 1:
                # RAINNC stores precipitation as a continuously increasing staircase - difference currently and previous stairstep to get true height of current
                # Calculate precipitation at current time step
                diff_vals = clean_ds[args.vnew].values - prev_ds[args.vnew].values

                # Put the resulting difference back into a copy of clean_ds
                diff_ds = clean_ds.copy()
                diff_ds[args.vnew] = (dims_3d, diff_vals)

                # Save new ds to netcdf
                diff_ds.to_netcdf(out_path)

                # Set prev_ds to current for the next file in the loop
                prev_ds = clean_ds

            elif args.vname == 'Q2':
                # Convert Q2 to specific humidity
                sph_ds = clean_ds / (1 + clean_ds)

                # Save new ds to netcdf
                sph_ds.to_netcdf(out_path)

            else:
                # Save new ds to netcdf
                clean_ds.to_netcdf(out_path)

            # Log file save success
            clean_file_paths.append(out_path)

# Print statement when finished
print(f'Successfully processed and saved {len(clean_file_paths)} clean files!')

# %%
