#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 13:19:23 2026

@author: u1301408
"""

import xarray as xr

wbt_open = xr.open_dataset('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/wbt/macav2metdata_GSLBIP_KACE-1-0-G_ssp585_wbt.nc')
wbt = wbt_open['wbt']

elevation_open = xr.open_dataset('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/elevation/macav2metdata_GSLBIP_KACE-1-0-G_ssp585_elevation.nc')
elevation = elevation_open['elevation']
slc_elevation = elevation.sel(lat = slice(40, 41.5), lon = slice(-113, -111))
# min = 1,279.26 m
# max = 3,133.91 m

# map of the Greater Salt Lake Area
slc = wbt.sel(lat = slice(40, 41.5), lon = slice(-113, -111))
wbt_winter = slc.sel(time = slc.time.dt.month.isin([12, 1, 2])) # only winter months

def snow_fraction_band(min_elevation, max_elevation):
    
    # elevation bands of every 300m starting at 1,200
    mask = (slc_elevation >= min_elevation) & (slc_elevation <= max_elevation)
    wbt_masked = wbt_winter.where(mask)
    
    # condition, if true, if false
    chanceOrain = xr.where(wbt_masked >= 0.5, 1, 0) # 1 is 100% chance of rain -> over 0.5 C wbt
    
    data = []
    for year in range(1979, 2100):
        year_mask = [t.year == year for t in chanceOrain.time.values]
        yearly = chanceOrain.sel(time = year_mask)
        rain_days = yearly.sum(dim = "time") # the total number of rain days that year
        
        # what fraction of the winter days that decade are cold enough to snow
        snow_fraction = (yearly.count(dim = "time") - rain_days )/ yearly.count(dim = "time")
        data.append(float(snow_fraction.mean()))
    
    return data # larger value means more snow


first = snow_fraction_band(1200, 1500)
second = snow_fraction_band(1500, 1800)
third = snow_fraction_band(1800, 2100)
fourth = snow_fraction_band(2100, 2400)
fifth = snow_fraction_band(2400, 2700)
sixth = snow_fraction_band(2700, 3000)
seventh = snow_fraction_band(3000, 3300)






        
    
    
    