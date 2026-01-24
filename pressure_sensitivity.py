#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:58:27 2026

@author: u1301408
"""

import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt


### Open temperature data, select winter months (DJF), and calculate daily mean ###
min_fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_ACCESS-CM2_ssp585_tasmin.nc'
min_open = xr.open_dataset(min_fpath)
tasmin_ds = min_open['tasmin'].sel(time = min_open['tasmin'].time.dt.month.isin([12, 1, 2]))

max_fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_ACCESS-CM2_ssp585_tasmax.nc'
max_open = xr.open_dataset(max_fpath)
tasmax_ds = max_open['tasmax'].sel(time = max_open['tasmax'].time.dt.month.isin([12, 1, 2]))

Temp_K = ((tasmax_ds + tasmin_ds) / 2)
# averages temperature and converts to C
Temp_C = Temp_K - 273.15 


### Open specific humidity data and select winter months (DJF) ###
huss_fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_ACCESS-CM2_ssp585_huss.nc'
huss_open = xr.open_dataset(huss_fpath)
spec_hum = huss_open['huss'].sel(time = huss_open['huss'].time.dt.month.isin([12, 1, 2]))


### Option #1: Constant pressure ###
# assuming constant pressure (hPa)
P_constant = 871.6 


### Option #2: Using surface temperature to derive surface pressure from sea level pressure ###
weight_ratio = 0.61 # kg dry air / kg water vapor
# mixing ratio (%)
mixing_ratio = spec_hum / (1 - spec_hum) # kg/kg
# virtual temperature
virtual_temp =  Temp_K(1 - (weight_ratio * mixing_ratio))
a = 29.3 # m / K
# elevation = 
# psl = 
# sea_level_pressure is the value from GCM data
P_surface_temp = psl * np.exp(elevation / (a * virtual_temp))


### Option #3: Using standard atmosphere to derive surface pressure from sea level pressure ###
# equation for the standard atmosphere is based on geopotential height and molecular scale temperature gradient (M)
# first calculated geopotential height (eq 18 and 19) and then use table 4 to get a value for M
# pressure can then be computed as a function of geopotential altitude using equation 33a
gamma = 1 # standard gravity / effective gravity is essentially 1
r_knot = 6356766 # effective radius (m)
# Z = # geometric height (m)
# geopotential height
geo_pot_hgt = gamma * ((r_knot * Z) / (r_knot + Z))



#%%
# plot rh as a function of P and visually see the sensitivity of the equation to different inputs
# analysis of percent change with different inputs

### Specific humidity to relative humidity conversion ###
# vapor pressure (hPa)
e = (spec_hum * P) / (0.622) # 0.622 is the constant that links pressure ratios to mass ratios
# Bolton 1980 Formule - saturation vapor pressure (hPa)
e_sub_s = 6.112 * np.exp((17.67 * Temp_C) / (Temp_C + 243.5))
# relative humidity as %
rh =  100 * (e / e_sub_s) 