#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:58:41 2026

@author: u1301408
"""

import matplotlib.pyplot as plt
import nc_time_axis
import sys
print(sys.executable)
import xarray as xr

model_name = 'KACE-1-0-G'
emission_scenario = 'ssp585'
start_year = 2030
stop_year  = 2040

wbt_open = xr.open_dataset(f'/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/wbt/macav2metdata_GSLBIP_{model_name}_{emission_scenario}_wbt.nc')
wbt = wbt_open['wbt']

snowbasin = wbt.sel(lat = 41.21, lon = -111.85, method = 'nearest')
snowbasin_winter = snowbasin.sel(time = snowbasin.time.dt.month.isin([12, 1, 2])) # only winter months

solider_hollow = wbt.sel(lat = 40.4667, lon = -111.5, method = 'nearest')
solider_hollow_winter = solider_hollow.sel(time = solider_hollow.time.dt.month.isin([12, 1, 2])) # only winter months


fig, ax = plt.subplots()

snowbasin_winter.plot(ax = ax, label = 'Snowbasin', linewidth = 0.5)
solider_hollow_winter.plot(ax = ax, label = 'Soldier Hollow', linewidth = 0.5)

ax.set_ylabel("Wet Bulb Temperature (°C)")
ax.legend
