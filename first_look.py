#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 16:06:53 2026

@author: u1301408
"""

import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs


min_fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_ACCESS-CM2_ssp585_tasmin.nc'
max_fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_ACCESS-CM2_ssp585_tasmax.nc'
huss_fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_ACCESS-CM2_ssp585_huss.nc'

min_open = xr.open_dataset(min_fpath)
tasmin_ds = min_open['tasmin'].sel(time = min_open['tasmin'].time.dt.month.isin([12, 1, 2]))

max_open = xr.open_dataset(max_fpath)
tasmax_ds = max_open['tasmax'].sel(time = max_open['tasmax'].time.dt.month.isin([12, 1, 2]))

Temp_K = ((tasmax_ds + tasmin_ds) / 2)

# averages temperature and converts to C
Temp_C = Temp_K - 273.15 

huss_open = xr.open_dataset(huss_fpath)
huss = huss_open['huss'].sel(time = huss_open['huss'].time.dt.month.isin([12, 1, 2]))

# specific humidity to relative humidity conversion assuming pressure = 101.325 kPa
P = 87.16 # SHOULDNT BE A CONSTANT PRESSURE
vapor_press = (huss * P) / (0.622 + huss) # 0.622 is the constant that links pressure ratios to mass ratios
# magnus formula
Psat = 0.61078 * np.exp((17.27 * Temp_C) / (Temp_C + 237.3)) # saturated vapor pressure in kPa
rh =  100 * (Psat / vapor_press) 




R = 8.3145 # J/K/mol ideal gas constant
M = 18 # g/mol molecular weight
rh_new = (huss * R * Temp_K) * (100 / (Psat / M))
#%%
# rain = np.where(temp_avg < 277.15, temp_avg, 0)
# temp_avg > 271.15

# 0.5C wet bulb temperature threshold

wbt = Temp_C * np.arctan( 0.151977 * ((rh + 8.313659)**(1/2)) ) + \
        np.arctan(Temp_C + rh) - np.arctan(rh - 1.676331) + \
            0.00391838 * ((rh)**(3/2)) * np.arctan(0.023101 * rh) - 4.686035
            
#%%            
slc = wbt.sel(lat = slice(40, 41.5), lon = slice(-113, -111))
wbt_set = slc.sel(time = slc.time.dt.year.isin(range(2020, 2030)))
wbt_avg = wbt_set.mean(dim = 'time')

# condition, if true, if false
# 1 is 100% chance of rain
chanceOrain = np.where(wbt_avg >= 0.5, 1, 0)
            
snowbasin_lat, snowbasin_lon = 41.21, -111.85
solider_hollow_lat, solider_hollow_lon = 40.4667, -111.5

# plot average 
fig, ax = plt.subplots(figsize=(8,5), subplot_kw = {'projection' : ccrs.PlateCarree()})

cf = ax.contourf(slc['lon'], slc['lat'], chanceOrain, levels=20, cmap='bwr', transform = ccrs.PlateCarree())
colorbar = plt.colorbar(cf, ax = ax, orientation = 'vertical')

states = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_1_states_provinces_lines', scale = '50m', facecolor = 'none', edgecolor = 'k', zorder = 4)
ax.add_feature(states, linewidth = 1.3)
ax.add_feature(cfeature.LAKES, zorder = 2, linewidth = 1)
ax.add_feature(cfeature.RIVERS, linewidth = 1, zorder = 1)

ax.plot(snowbasin_lon, snowbasin_lat, marker = '*', markersize = 15, color = 'black', transform = ccrs.PlateCarree())
ax.plot(solider_hollow_lon, solider_hollow_lat, marker = '*', markersize = 15, color = 'black', transform = ccrs.PlateCarree())

plt.show()

#%%










