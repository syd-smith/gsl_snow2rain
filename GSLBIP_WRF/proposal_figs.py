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


def SWE_fig(data):
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
        SWE_date = data.sel(Time = np.isin(SWE['Time.year'], years))

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

    return fig


def snow_fraction(SR_data, HGT_data):
    """
    Build inputs for pcolormesh function to show a model's change in snow_fraction 
    with elevation over time. A higher snow fraction means a higher percent chance
    of snow. Data is focused on the greater Salt Lake Area. 
    """
    
    # open a slice wet bulb data to only look at winter months around Salt Lake
    slc = SR_data.sel(lat = slice(40, 41.5), lon = slice(-113, -111))
    SR_winter = slc.sel(time = slc.time.dt.month.isin([12, 1, 2])) # only winter months
    
    # open and slice elevation data to only the focus area
    slc_elevation = HGT_data.sel(lat = slice(40, 41.5), lon = slice(-113, -111))

    snow_fractions=[]
    elevations = [1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300] # selected elevation bands
    
    for index in range(len(elevations) - 1):
        # mask data to only calculate snow_fraction in a specific elevation band
        min_elevation = elevations[index]
        max_elevation = elevations[index + 1]
        
        # elevation bands of every 300m starting at 1,200
        mask = (slc_elevation >= min_elevation) & (slc_elevation <= max_elevation)
        SR_masked = SR_winter.where(mask)
        
        # condition, if true, if false
        chanceOrain = xr.where(SR_masked >= 0.5, 1, 0) # 1 is 100% chance of rain -> over 0.5C wbt
        
        data = []
        for year in range(1979, 2100):
            if chanceOrain.time.dtype == 'O':
                year_mask = [t.year == year for t in chanceOrain.time.values]
                yearly = chanceOrain.sel(time = year_mask)
            else:
                yearly  = chanceOrain.sel(time = chanceOrain.time.dt.year == year)
                
            rain_days = yearly.sum(dim = "time") # the total number of rain days that year
            
            # what fraction of the winter days that decade are cold enough to snow 
            # (number of total days - number of rain days) / number of total days
            snow_fraction = (yearly.count(dim = "time") - rain_days )/ yearly.count(dim = "time")
            # average snow_fraction over that elevation for that year
            data.append(float(snow_fraction.mean()))
            
        # larger value means more snow
        snow_fractions.append(data)
        
        times = list(range(1979, 2101))
        fig, ax = plt.subplots(1, 1, figsize = (20, 5))

    # arrange levels by 10% intervals
    levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # Olympics set snow threshold at having snow 90% of the time?
    norm = TwoSlopeNorm(vmin = 0, vmax = 1, vcenter = 0.9) 

    data = ax.pcolormesh(times, elevations, snow_fractions, shading = 'auto', cmap  = cmap.cmap('MPL_bwr', revBool = True), norm = norm)
    cbar = fig.colorbar(data, ax = ax, orientation = 'vertical', norm = norm, ticks = levels, pad = 0.02, extend = 'both', boundaries = levels)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    ax.set_yticks(elevations)
    ax.set_xticks([1980, 1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100])
    ax.set_ylabel('Elevation (m)')
    ax.axhline(y = 2210, color = 'red') # snowbasin elevation
    ax.axhline(y  =  2014, color = 'red') # solider hollow elevation
    
    return snow_fractions, elevations, times







#%%
# run data
# point to file
fpath = str(current_file_directory) / 'data_pull' 
# open SWE data
SWE = xr.open_dataset(fpath / 'SWE_month_slice.nc')['SWE']
# apply water day filter to data -> organizing data by water year
SWE_month_slice['water_day'] = get_water_day(SWE['Time'])

SR = xr.open_dataset(fpath / 'SR_month_slice.nc')['SR']
SR_month_slice['water_day'] = get_water_day(SR['Time'])


    # %%
