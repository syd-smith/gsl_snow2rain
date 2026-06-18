# %%
"""
Author: Sydney Smith
Date: June 17, 2026
"""

import cartopy.crs as ccrs
import glob
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
from wrf import (getvar, ALL_TIMES)
import xarray as xr

# ==================================
# - Establish Relative File Path - 
# ==================================

try:
    current_file_directory = Path(__file__).resolve().parent
except NameError:
    current_file_directory = Path().resolve().parent
parent_directory = current_file_directory.parent
sys.path.append(str(parent_directory))

import from_savanna.nclcmaps as cmap

# point to file
fpath = str(current_file_directory) / 'SWE_month_slice.nc'

# open SWE data
SWE = xr.open_dataset(fpath)['SWE']

# organize data to be inline with the water year (Oct 1 - April 1 instead of Jan 1 - April 1 then Oct 1 - Dec 31)
def get_water_day(time):
    # pull month and day from time variable
    month = time.dt.month
    day = time.dt.day
    
    # create offset for each month so that October is first and April is last
    month_offset = {10: 0, 11: 31, 12: 61, 1: 92, 2: 123, 3: 151, 4: 182}
    
    # apply offset
    offset = np.array([month_offset[m] for m in month.values])
    
    # adjust for leap year (if month > 2, add 1 day)
    is_leap = time.dt.is_leap_year
    offset += np.where((is_leap) & (month > 2), 1, 0)
    
    return day + offset

# apply water day filter to data -> organizing data by water year
SWE_month_slice['water_day'] = get_water_day(SWE['Time'])


# initialize figure
fig = plt.figure(figsize = (10, 10))

axes = []
for ax in range(4):
    # creates four rows of timeseries subplots with no projection
    ax_T = fig.add_subplot(4, 2, ax * 2 + 1)

    # creates four rows of map subplots with projection
    ax_M = fig.add_subplot(4, 2, ax * 2 + 2, projection = ccrs.PlateCarree())
    axes.append([ax_T, ax_M])

# define labels for each subplot
labels = [['a.', 'b,'],
          ['c.', 'd.'], 
          ['e.', 'f.'], 
          ['g.', 'h.']]

# define levels for map countouring
levels = np.linspace(0, 1200, 12) 

# define colormap for map contouring
colors = cmap.cmap('perc2_9lev', revBool = True)
blue = colors([0.4])
green = colors([0.5])
red = colors([0.9])

# define norm values to standardize colorbar to map colors
norm = mcolors.BoundaryNorm(levels, ncolors = plt.get_cmap(cmap, 14).N)   

# define time ranges for each row
time_ranges = [range(1985, 2015), range(2030, 2054), range(2054, 2078), range(2078, 2100)]

for idx, years in enumerate(time_ranges):
    # iterate on subplots created above - this pulls the timeseries and map axes for the same row
    ax_T, ax_M = axes[idx]
    pull_labels = labels[idx]

    # TIMESERIES PLOT
    # select years of interest for given subplot
    SWE_date = SWE_month_slice.sel(Time = np.isin(SWE['Time.year'], years))

    # location of snotel site near alta (Atwater)
    # Latitude: 40 deg; 35 min N
    # Longitude: 111 deg; 38 min W
    # select SWE values at a specific location (SLCDPU snotel site near Alta - Atwater)
    SWE_location = SWE_date.sel(XLONG = -111.6333, XLAT = 40.5833, method = 'nearest')

    # calculate the daily mean, min, and max SWE values for Atwater site
    swe_daily_mean = SWE_location.groupby('Time.dayofyear').mean(dim = 'Time')
    swe_daily_min = SWE_location.groupby('Time.dayofyear').min(dim = 'Time')
    swe_daily_max = SWE_location.groupby('Time.dayofyear').max(dim = 'Time')

    # plot time series of SWE data in the left column
    ax_T.plot(swe_daily_mean['Time'], swe_daily_mean.values, linestyle = '--', color = green)
    ax_T.plot(swe_daily_min['Time'], swe_daily_min.values, linestyle = '--', color = red)
    ax_T.plot(swe_daily_max['Time'], swe_daily_max.values, linestyle = '--', color = blue)

    # set tick labels
    ax_T.set_xticks([0, 31, 61, 92, 123, 151, 182])
    ax_T.set_xticklabels(['Oct 1', 'Nov 1', 'Dec 1', 'Jan 1', 'Feb 1', 'Mar 1', 'April 1'])
    ax_T.set_yticks(levels)
    ax_T.set_ylabels('SWE (kg m-2)')

    # MAP PLOT
    # set map extend
    extent = [float(lons[0][0]), float(lons[0][-1]), float(lats[2][0]), float(lats[-1][0])]
    ax_M.set_extent(extent, crs = ccrs.PlateCarree())
   
    # take mean SWE over the entire time period
    SWE_mean = SWE_date.mean(dim = 'Time')

    # plot map of average SWE over study region in right column
    contour = ax_M.contourf(SWE_mean['XLONG'], SWE_mean['XLAT'], SWE_mean, cmap = colors, norm = norm, transform = ccrs.PlateCarree())

    # add map features to orient within the map
    states = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_1_states_provinces_lines', scale = '50m', facecolor = 'none', edgecolor = 'k', zorder = 4)
    ax_M.add_feature(states, linewidth = 1.3)
    ax_M.add_feature(cfeature.LAKES, zorder = 2, linewidth = 1)
    ax_M.add_feature(cfeature.RIVERS, linewidth = 1, zorder = 1)

    # add labels to subplots
    ax_T.text(0.05, 0.9, pull_labels[0], transform = ax_T.transAxes, fontsize = 10, fontweight = 'bold')
    ax_M.text(0.05, 0.9, pull_labels[1], transform = ax_M.transAxes, fontsize = 10, fontweight = 'bold')

# create colorbar for the entire figure
fig.colorbar(contour, ax = axes, orientation = 'vertical', pad = 0.04, label = 'SWE (kg m-2)')

# create legend for entire figure

# %%
