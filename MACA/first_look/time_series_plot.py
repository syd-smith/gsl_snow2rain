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

wbt_open = xr.open_dataset(f'/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/wbt/macav2metdata_GSLBIP_{model_name}_{emission_scenario}_wbt.nc')
wbt = wbt_open['wbt']

snowbasin = wbt.sel(lat = 41.21, lon = -111.85, method = 'nearest')
snowbasin_winter = snowbasin.sel(time = snowbasin.time.dt.month.isin([12, 1, 2])) # only winter months

solider_hollow = wbt.sel(lat = 40.4667, lon = -111.5, method = 'nearest')
solider_hollow_winter = solider_hollow.sel(time = solider_hollow.time.dt.month.isin([12, 1, 2])) # only winter months


fig, axs = plt.subplots(2, 1, figsize = (10, 8))

# plot data using xarray.dataarray.plot
snowbasin_winter.plot(ax = axs[0], label = 'Snowbasin', linewidth = 0.5, color = 'blue')
axs[0].set_title('Snowbasin')
solider_hollow_winter.plot(ax = axs[1], label = 'Soldier Hollow', linewidth = 0.5, color = 'green')
axs[1].set_title('Solider Hollow')

# choose tick positions manually
ticks = [t for t in snowbasin_winter.time.values if t.year % 10 == 0 and t.month == 1 and t.day == 1]

# convert ticks to strings for labels
labels = [str(t.year) for t in ticks]

for ax in axs:
    print(ax)
    ax.set_ylabel("Wet Bulb Temperature (°C)")
    
    # adjust y labels to highlight 0.5C wbt transition point of snow to rain
    ax.set_yticks([-20, -15, -10, -5, 0.5, 5, 10])
    for label in ax.get_yticklabels():
        if label.get_text() == '0.5':
            label.set_color('red')
        else:
            label.set_color('black')
    ax.axhline(y = 0.5, linestyle = '--', color = 'red', linewidth = 0.75)

    # apply manual tick labels
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel('')


