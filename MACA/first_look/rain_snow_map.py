#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:54:20 2026

@author: u1301408
"""

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import xarray as xr

sys.path.append('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/GSLBIP/from_savanna/')
import nclcmaps as cmap


def map_snow(model_name, emission_scenario, start_year, stop_year):

    wbt_open = xr.open_dataset(f'/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/wbt/macav2metdata_GSLBIP_{model_name}_{emission_scenario}_wbt.nc')
    wbt = wbt_open['wbt']
    
    # map of the Greater Salt Lake Area
    slc = wbt.sel(lat = slice(40, 41.5), lon = slice(-113, -111))
    wbt_winter = slc.sel(time = slc.time.dt.month.isin([12, 1, 2])) # only winter months
    wbt_of_interest = wbt_winter.sel(time = wbt_winter.time.dt.year.isin(range(start_year, stop_year+1)))
    
    # condition, if true, if false
    # 1 is 100% chance of rain -> over 0.5 C wbt
    chanceOrain = xr.where(wbt_of_interest >= 0.5, 1, 0)
    rain_days = chanceOrain.sum(dim = "time") # the total number of rain days that decade
    # rain_fraction = rain_days / chanceOrain.count(dim = "time")
    # what fraction of the winter days that decade are cold enough to snow
    snow_fraction = (chanceOrain.count(dim = "time") - rain_days )/ chanceOrain.count(dim = "time")
    
    fig, ax = plt.subplots(figsize = (8,5), subplot_kw = {'projection' : ccrs.PlateCarree()})
    
    # standardize colorbar across all maps
    levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    norm = mcolors.BoundaryNorm(levels, ncolors = plt.get_cmap(cmap.cmap('hotres', revBool = True), 8).N)
    
    # map snow fraction data and add colorbar
    cf = ax.contourf(slc['lon'], slc['lat'], snow_fraction, levels = levels, cmap = cmap.cmap('hotres', revBool = True), transform = ccrs.PlateCarree())
    cbar = fig.colorbar(cf, ax = ax, orientation = 'vertical', norm = norm, ticks = levels, pad = 0.02, extend = 'both', boundaries = levels)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    
    # add map features
    states = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_1_states_provinces_lines', scale = '50m', facecolor = 'none', edgecolor = 'k', zorder = 4)
    ax.add_feature(states, linewidth = 1.3)
    ax.add_feature(cfeature.LAKES, zorder = 2, linewidth = 1)
    ax.add_feature(cfeature.RIVERS, linewidth = 1, zorder = 1)
    
    # mark areas of significance to the Olympic Games
    snowbasin_lat, snowbasin_lon = 41.21, -111.85
    solider_hollow_lat, solider_hollow_lon = 40.4667, -111.5
    ax.plot(snowbasin_lon, snowbasin_lat, marker = '*', markersize = 15, color = 'black', transform = ccrs.PlateCarree())
    ax.plot(solider_hollow_lon, solider_hollow_lat, marker = '*', markersize = 15, color = 'black', transform = ccrs.PlateCarree())
    
    plt.show()
    
    return fig, ax

face, bones = map_snow('KACE-1-0-G', 'ssp585', 2030, 2040)