#%%
"""
Created:
By: Court Strong, court.strong@utah.edu

Edited: October 1st, 2024
Edited: July 29th, 2024
By: Savanna Wolvin, s.wolvin@utah.edu

Edited: October 27, 2026
By: Sydney Smith

# SUMMARY
Calculate the maximum or minimum possible daily mean solar radiation on the 
coarse scale and fine scale grids, then save values to MATLAB files.

# INPUT
solar_constant

# GLOBAL INPUTS FROM PARAMETERS.JSON
    GCM_DIR     -> Directory to the GCM data in MATLAB files of coarse resolution
    OUTPUT_DIR  -> Directory to the GCM data in MATLAB files of fine resolution
    RSDS_DIR    -> Directory to save the solar radiation data

# OUTPUTS
    {RSDS_DIR}max_rsds_coarse.mat 
        MATLAB file containing the daily maximum solar radiation on the coarse grid
    
    {RSDS_DIR}min_rsds_coarse.mat
        MATLAB file containing the daily minimum solar radiation on the coarse grid
    
    {RSDS_DIR}max_rsds_fine.mat
        MATLAB file containing the daily maximum solar radiation on the fine grid
    
    {RSDS_DIR}min_rsds_fine.mat
        MATLAB file containing the daily minimum solar radiation on the fine grid
"""

#Python libraries and variables

# Incoming solar radiation (W m-2)
solar_constant = 1361

# Units
ux = 'W m-2'

import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import glob
# import h5py
import json
import xarray as xr

# # Load in global parameters
# with open(f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "../01_parameter_settings/"))}/PARAMETERS.json', 'r') as f:
#     params = json.load(f)

# GCM_DIR    = params['GCM_DIR']
# OUTPUT_DIR = params['OUTPUT_DIR']
# RSDS_DIR   = params['RSDS_DIR']

"""
FUNCTION DEFINITION: calculate_daily_solar_radiation
    INPUT
    - lat (lat x lon): Latitude mesh grid as array
    - lon (lat x lon): Longitude mesh grid as array
    - day_of_year (int): Day of year, needed to formulate the declination angle
    - cloud_cover_fraction (float): Ranges between 0.0 and 1.0, used to 
            formulate the reduction in solar radiation due to clouds
    
    OUTPUT
    - daily_mean_radiation (lat x lon): Daily radiation value on a latitude- 
            longitude grid based on the day of year and cloud cover fraction

"""
def calculate_daily_solar_radiation(lat, lon, day_of_year, cloud_cover_fraction=0):
    """
    Calculate the maximum or minimum possible daily mean solar radiation on a grid for a given day of the year.
    The cloud_cover_fraction parameter adjusts the radiation for cloud cover.
    """
    # Constants
    # solar_constant = 1361  # W/m^2

    # Convert latitude to radians
    lat_rad = np.radians(lat)
    
    # Calculate the solar declination angle (approximation)
    declination = 23.45 * np.sin(np.radians((360/365.25) * (day_of_year - 81)))
    print(declination)

    # Convert declination to radians
    declination_rad = np.radians(declination)
    print(declination_rad)

    # Calculate the solar hour angle at each point in the day
    hour_angles = np.linspace(-np.pi, np.pi, 1000)  # 1000 points for integration
    
    # Integrate the daily solar radiation over the solar hour angle
    cos_zenith_angle = (np.sin(lat_rad) * np.sin(declination_rad) +
                        np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angles[:, np.newaxis, np.newaxis]))
    print(cos_zenith_angle)

    # Calculate the instantaneous solar radiation
    instantaneous_radiation = solar_constant * np.maximum(cos_zenith_angle, 0)

    # Apply cloud cover reduction (e.g., 85% reduction means 15% passes through)
    instantaneous_radiation *= (1 - cloud_cover_fraction * 0.85)

    # Integrate over the hour angle
    # np.trapz was depreciated
    integrated_radiation = np.trapezoid(instantaneous_radiation, hour_angles, axis=0)

    # Convert to daily mean solar radiation in W/m^2
    daily_mean_radiation = integrated_radiation / (2 * np.pi)

    return daily_mean_radiation



# if __name__ == "__main__":
#     # Load the example .mat file to get the coarse lat and lon grid
#     files = glob.glob(os.path.join(GCM_DIR, 'ACCESS*.mat'))
    
#     if len(files) == 0:
#         raise FileNotFoundError("No files matching 'ACCESS*.mat' found in the specified directory.")
    
#     gcm_example = loadmat(files[0])
    
#     COARSE_LAT = gcm_example['lat'].flatten()
#     COARSE_LON = np.mod(gcm_example['lon'].flatten() + 180, 360) - 180  # wrapTo180 equivalent
    
#     COARSE_LAT_MESH, COARSE_LON_MESH = np.meshgrid(COARSE_LAT, COARSE_LON, indexing='ij')  # Ensure correct indexing
    
#     # Load the fine lat and lon grid
#     fine_files = glob.glob(os.path.join(OUTPUT_DIR, '*ACCESS*.mat'))
    
#     if len(fine_files) == 0:
#         raise FileNotFoundError("No files matching 'FINE_GRID*.mat' found in the specified directory.")
    
#     fine_example = fine_files[0]
    
#     with h5py.File(fine_example, 'r') as f:
#         FINE_LAT = f['lat'][:].flatten()
#         FINE_LON = np.mod(f['lon'][:].flatten() + 180, 360) - 180  # wrapTo180 equivalent
    
#     FINE_LAT_MESH, FINE_LON_MESH = np.meshgrid(FINE_LAT, FINE_LON, indexing='ij')  # Ensure correct indexing
    
#     # Initialize arrays to store solar radiation data for each day
#     max_solar_radiation_all_days_coarse = np.zeros((len(COARSE_LAT), len(COARSE_LON), 366))
#     min_solar_radiation_all_days_coarse = np.zeros((len(COARSE_LAT), len(COARSE_LON), 366))
#     max_solar_radiation_all_days_fine = np.zeros((len(FINE_LAT), len(FINE_LON), 366))
#     min_solar_radiation_all_days_fine = np.zeros((len(FINE_LAT), len(FINE_LON), 366))
    
#     # Loop through each day of the year (1 to 366)
#     for day_of_year in range(1, 367):
#         max_solar_radiation_coarse = calculate_daily_solar_radiation(COARSE_LAT_MESH, COARSE_LON_MESH, day_of_year)
#         min_solar_radiation_coarse = calculate_daily_solar_radiation(COARSE_LAT_MESH, COARSE_LON_MESH, day_of_year, cloud_cover_fraction=1.0)  # 100% cloud cover
#         max_solar_radiation_fine = calculate_daily_solar_radiation(FINE_LAT_MESH, FINE_LON_MESH, day_of_year)
#         min_solar_radiation_fine = calculate_daily_solar_radiation(FINE_LAT_MESH, FINE_LON_MESH, day_of_year, cloud_cover_fraction=1.0)  # 100% cloud cover
    
#         max_solar_radiation_all_days_coarse[:, :, day_of_year-1] = max_solar_radiation_coarse
#         min_solar_radiation_all_days_coarse[:, :, day_of_year-1] = min_solar_radiation_coarse
#         max_solar_radiation_all_days_fine[:, :, day_of_year-1] = max_solar_radiation_fine
#         min_solar_radiation_all_days_fine[:, :, day_of_year-1] = min_solar_radiation_fine
    
#         # Generate and display the map only for the first day
#         if day_of_year == 1:
#             plt.figure(figsize=(12, 6))
#             ax = plt.axes(projection=ccrs.PlateCarree())
#             ax.coastlines()
#             contour = ax.contourf(COARSE_LON, COARSE_LAT, max_solar_radiation_coarse, 60, transform=ccrs.PlateCarree(), cmap='jet')
#             plt.colorbar(contour, label='Daily Mean Solar Radiation (W/m^2)')
#             plt.title(f'Day {day_of_year} - Maximum Daily Mean Solar Radiation (Coarse)')
#             plt.show()
    
#             plt.figure(figsize=(12, 6))
#             ax = plt.axes(projection=ccrs.PlateCarree())
#             ax.coastlines()
#             contour = ax.contourf(COARSE_LON, COARSE_LAT, min_solar_radiation_coarse, 60, transform=ccrs.PlateCarree(), cmap='jet')
#             plt.colorbar(contour, label='Daily Mean Solar Radiation (W/m^2)')
#             plt.title(f'Day {day_of_year} - Minimum Daily Mean Solar Radiation (Full Cloud Cover) (Coarse)')
#             plt.show()
            
#             plt.figure(figsize=(12, 6))
#             ax = plt.axes(projection=ccrs.PlateCarree())
#             ax.coastlines()
#             contour = ax.contourf(FINE_LON, FINE_LAT, max_solar_radiation_fine, 60, transform=ccrs.PlateCarree(), cmap='jet')
#             plt.colorbar(contour, label='Daily Mean Solar Radiation (W/m^2)')
#             plt.title(f'Day {day_of_year} - Maximum Daily Mean Solar Radiation (Fine)')
#             plt.show()
    
#             plt.figure(figsize=(12, 6))
#             ax = plt.axes(projection=ccrs.PlateCarree())
#             ax.coastlines()
#             contour = ax.contourf(FINE_LON, FINE_LAT, min_solar_radiation_fine, 60, transform=ccrs.PlateCarree(), cmap='jet')
#             plt.colorbar(contour, label='Daily Mean Solar Radiation (W/m^2)')
#             plt.title(f'Day {day_of_year} - Minimum Daily Mean Solar Radiation (Full Cloud Cover) (Fine)')
#             plt.show()
    
#     max_solar_radiation_all_days_fine = max_solar_radiation_all_days_fine.astype(np.float32)
#     min_solar_radiation_all_days_fine = min_solar_radiation_all_days_fine.astype(np.float32)
    
#     # Save the data to separate .mat files
#     savemat(f'{RSDS_DIR}max_rsds_coarse.mat', 
#             {'lat': COARSE_LAT, 'lon': COARSE_LON, 'max_solar_radiation': max_solar_radiation_all_days_coarse, 'units': ux})
#     savemat(f'{RSDS_DIR}min_rsds_coarse.mat', 
#             {'lat': COARSE_LAT, 'lon': COARSE_LON, 'min_solar_radiation': min_solar_radiation_all_days_coarse, 'units': ux})
#     savemat(f'{RSDS_DIR}max_rsds_fine.mat', 
#             {'lat': FINE_LAT, 'lon': FINE_LON, 'max_solar_radiation': max_solar_radiation_all_days_fine.astype(np.float32), 'units': ux})
#     savemat(f'{RSDS_DIR}min_rsds_fine.mat', 
#             {'lat': FINE_LAT, 'lon': FINE_LON, 'min_solar_radiation': min_solar_radiation_all_days_fine.astype(np.float32), 'units': ux})
    
#     print("Data saved successfully.")