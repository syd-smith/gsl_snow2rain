#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:14:26 2026

@author: u1301408
"""

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr


### Open temperature data, select winter months (DJF), and calculate daily mean ###
min_fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_ACCESS-CM2_ssp585_tasmin.nc'
min_open = xr.open_dataset(min_fpath)
tasmin = min_open['tasmin']
# tasmin_DJF = min_open['tasmin'].sel(time = min_open['tasmin'].time.dt.month.isin([12, 1, 2]))

max_fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_ACCESS-CM2_ssp585_tasmax.nc'
max_open = xr.open_dataset(max_fpath)
tasmax = max_open['tasmax']
# tasmax_DJF = max_open['tasmax'].sel(time = max_open['tasmax'].time.dt.month.isin([12, 1, 2]))

Temp_K = ((tasmax + tasmin) / 2)
# averages temperature and converts to C
Temp_C = Temp_K - 273.15 

### Open specific humidity data and select winter months (DJF) ###
huss_fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_ACCESS-CM2_ssp585_huss.nc'
huss_open = xr.open_dataset(huss_fpath)
spec_hum  = huss_open['huss'] # units: kg/kg
# spec_hum_DJF = huss_open['huss'].sel(time = huss_open['huss'].time.dt.month.isin([12, 1, 2]))
spec_hum = spec_hum.load()

### elevation data ###
elevation_open = xr.open_dataset('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/gridmet_interp_elevation.nc')
elevation_loaded = elevation_open['elevation'].load()
elevation_corrected = elevation_loaded.sortby('lat', ascending = True) # reverses the order of lat
elevation_assigned = elevation_corrected.assign_coords(lat = spec_hum.lat, lon = spec_hum.lon)
elevation = elevation_assigned.expand_dims(time = spec_hum.time) # units: meters

### calculate pressure based on elevation data ###
Ps = 1013.25 # sea level pressure (hPa)
H = 7600 # scale height (m)
P = Ps * np.exp(-elevation / H)
# P = P.load()

### Specific humidity to relative humidity conversion ###
# vapor pressure (hPa)
e = (spec_hum * P) / (0.622) # 0.622 is the constant that links pressure ratios to mass ratios
# Bolton 1980 Formule - saturation vapor pressure (hPa)
e_sub_s = 6.112 * np.exp((17.67 * Temp_C) / (Temp_C + 243.5))
# relative humidity as %
rh =  100 * (e / e_sub_s) 

### wet bulb temperature calculation ###
wbt = Temp_C * np.arctan( 0.151977 * ((rh + 8.313659)**(1/2)) ) + \
        np.arctan(Temp_C + rh) - np.arctan(rh - 1.676331) + \
            0.00391838 * ((rh)**(3/2)) * np.arctan(0.023101 * rh) - 4.686035
            
#%%
wbt.to_netcdf('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/wbt.nc')

#%%
slc = wbt.sel(lat = slice(40, 41.5), lon = slice(-113, -111))
wbt_winter = slc.sel(time = slc.time.dt.month.isin([12, 1, 2]))
wbt_of_interest = wbt_winter.sel(time = wbt_winter.time.dt.year.isin(range(2030, 2040)))
wbt_avg = wbt_of_interest.mean(dim = 'time')


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
