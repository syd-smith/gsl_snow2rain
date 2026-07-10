#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 13:19:23 2026

@author: u1301408
"""

import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import xarray as xr

sys.path.append('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/GSLBIP/from_savanna/')
import nclcmaps as cmap


def snow_fraction(model, emission_scenario):
    """
    Build inputs for pcolormesh function to show a model's change in snow_fraction 
    with elevation over time. A higher snow fraction means a higher percent chance
    of snow. Data is focused on the greater Salt Lake Area. 
    """
    
    # open a slice wet bulb data to only look at winter months around Salt Lake
    wbt_open = xr.open_dataset(f'/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/wbt/macav2metdata_GSLBIP_{model}_{emission_scenario}_wbt.nc')
    wbt = wbt_open['wbt']
    slc = wbt.sel(lat = slice(40, 41.5), lon = slice(-113, -111))
    wbt_winter = slc.sel(time = slc.time.dt.month.isin([12, 1, 2])) # only winter months
    
    # open and slice elevation data to only the focus area
    elevation_open = xr.open_dataset(f'/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/elevation/macav2metdata_GSLBIP_{model}_{emission_scenario}_elevation.nc')
    elevation = elevation_open['elevation']
    slc_elevation = elevation.sel(lat = slice(40, 41.5), lon = slice(-113, -111))

    snow_fractions=[]
    elevations = [1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300] # selected elevation bands
    
    for index in range(len(elevations) - 1):
        # mask data to only calculate snow_fraction in a specific elevation band
        min_elevation = elevations[index]
        max_elevation = elevations[index + 1]
        
        # elevation bands of every 300m starting at 1,200
        mask = (slc_elevation >= min_elevation) & (slc_elevation <= max_elevation)
        wbt_masked = wbt_winter.where(mask)
        
        # condition, if true, if false
        chanceOrain = xr.where(wbt_masked >= 0.5, 1, 0) # 1 is 100% chance of rain -> over 0.5C wbt
        
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
    
    return snow_fractions, elevations, times

snow_fractions, elevations, times = snow_fraction('KACE-1-0-G', 'ssp585')

#%%
fig, ax = plt.subplots(1, 1, figsize = (20, 5))

levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
norm = TwoSlopeNorm(vmin = 0, vmax = 1, vcenter = 0.9) # Olympics set snow threshold at 90% of the time?

data = ax.pcolormesh(times, elevations, snow_fractions, shading = 'auto', cmap  = cmap.cmap('MPL_bwr', revBool = True), norm = norm)
cbar = fig.colorbar(data, ax = ax, orientation = 'vertical', norm = norm, ticks = levels, pad = 0.02, extend = 'both', boundaries = levels)
cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

ax.set_yticks(elevations)
ax.set_xticks([1980, 1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100])
ax.set_ylabel('Elevation (m)')
ax.axhline(y = 2210, color = 'red') # snowbasin elevation
ax.axhline(y  =  2014, color = 'red') # solider hollow elevation








        
    
    
    