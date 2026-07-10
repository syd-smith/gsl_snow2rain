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

#%%
### Open sea level pressure files ###
psl_open = xr.open_mfdataset(['/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/press_data/psl_day_ACCESS-CM2_historical_r1i1p1f1_gn_19500101-19990706.nc',
                         '/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/press_data/psl_day_ACCESS-CM2_historical_r1i1p1f1_gn_19990707-20141231.nc',
                         '/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/press_data/psl_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_20150101-20640705.nc',
                         '/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/press_data/psl_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_20640706-20991231.nc'],
                        combine = "by_coords")
# temp time dimensions: 1979-01-01 1979-01-02 ... 2099-12-31
psl_select =  psl_open['psl']
psl_time_slice = psl_select.sel(time = slice('1979-01-01', '2099-12-31')) # units: Pa

def convert_lon_to_360(lon):
    
    """
    GCM lon values are in 0-360 format while other datasets use -180-180 -> 
    conversion may be needed.
    """
    
    # Convert longitude from -180-180 to 0-360
    lon = np.array(lon)
    lon_360 = (lon + 360) % 360
    
    return lon_360

def convert_lon_360_to_180(lon):
    
    """
    Convert longitude from 0360 range to -180 to 180 range
    """
    
    lon = np.array(lon)  # Ensures input works for lists or arrays
    lon_180 = ((lon + 180) % 360) - 180
    
    return lon_180

psl_single = psl_time_slice.sel(lat = 40.6226, lon = convert_lon_to_360(-111.4851), method = 'nearest')
psl_broadcast = xr.broadcast(psl_single, Temp_K)[0]
# psl = psl_expanded * xr.ones_like(Temp_K)

#%%
### elevation data ###
elevation_open = xr.open_dataset('/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/gridmet_interp_elevation.nc')
elevation = elevation_open['elevation'].expand_dims(time = Temp_K.time) # units: meters
elevation_corrected = elevation.sel(lat = elevation.lat[::-1])

#%%
### Option #1: Constant pressure ###
# assuming constant pressure (hPa)
P_constant = 871.6 


### Option #2: Using surface temperature to derive surface pressure from sea level pressure ###
weight_ratio = 0.61 # kg dry air / kg water vapor
# mixing ratio (%)
mixing_ratio = spec_hum / (1 - spec_hum) # kg/kg
# virtual temperature
virtual_temp =  Temp_K * (1 - (weight_ratio * mixing_ratio))
a = 29.3 # m / K
# sea_level_pressure is the value from GCM data
P_surface_temp = psl_broadcast * np.exp(elevation / (a * virtual_temp))

#%%
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
P = 870 

### Specific humidity to relative humidity conversion ###
# vapor pressure (hPa)
e = (spec_hum * P) / (0.622) # 0.622 is the constant that links pressure ratios to mass ratios
# Bolton 1980 Formule - saturation vapor pressure (hPa)
e_sub_s = 6.112 * np.exp((17.67 * Temp_C) / (Temp_C + 243.5))
# relative humidity as %
rh =  100 * (e / e_sub_s) 

#%%
# plot rh as a function of P and visually see the sensitivity of the equation to different inputs
# analysis of percent change with different inputs
P1 = 770

### Specific humidity to relative humidity conversion ###
# vapor pressure (hPa)
e1 = (spec_hum * P1) / (0.622) # 0.622 is the constant that links pressure ratios to mass ratios
# Bolton 1980 Formule - saturation vapor pressure (hPa)
e_sub_s1 = 6.112 * np.exp((17.67 * Temp_C) / (Temp_C + 243.5))
# relative humidity as %
rh1 =  100 * (e1 / e_sub_s1) 

#%%
rh = rh.sel(lat=42, lon=-111, method='nearest')
rh_low = rh1.sel(lat=42, lon=-111, method='nearest')

plt.plot(rh.time, rh, label='870 hPa')
plt.plot(rh_low.time, rh_low, label='770 hPa')
plt.legend()
plt.show()


#%%
location = spec_hum.sel(lat = 42, lon = -111,  method = 'nearest')
loc_avg = location.mean(dim = 'time')
#%%
Temp_C_loc = Temp_C.sel(lat = 42, lon = -111,  method = 'nearest')
temp_avg = Temp_C_loc.mean(dim = 'time')

#%%
# plot rh as a function of P and visually see the sensitivity of the equation to different inputs
# analysis of percent change with different inputs
P = 790

### Specific humidity to relative humidity conversion ###
# vapor pressure (hPa)
e = (loc_avg * P) / (0.622) # 0.622 is the constant that links pressure ratios to mass ratios
# Bolton 1980 Formule - saturation vapor pressure (hPa)
e_sub_s = 6.112 * np.exp((17.67 * temp_avg) / (temp_avg + 243.5))
# relative humidity as %
rh_samp =  100 * (e / e_sub_s) 
