#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:14:26 2026

@author: u1301408
"""

import numpy as np
import xarray as xr


def wbt_press_calc(model_name, emission_scenario):
    
    ### Open temperature data, select winter months (DJF), and calculate daily mean ###
    min_fpath = f'/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_{model_name}_{emission_scenario}_tasmin.nc'
    min_open = xr.open_dataset(min_fpath)
    tasmin = min_open['tasmin']
    # tasmin_DJF = min_open['tasmin'].sel(time = min_open['tasmin'].time.dt.month.isin([12, 1, 2]))
    
    max_fpath = f'/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_{model_name}_{emission_scenario}_tasmax.nc'
    max_open = xr.open_dataset(max_fpath)
    tasmax = max_open['tasmax']
    # tasmax_DJF = max_open['tasmax'].sel(time = max_open['tasmax'].time.dt.month.isin([12, 1, 2]))
    
    Temp_K = ((tasmax + tasmin) / 2)
    # averages temperature and converts to C
    Temp_C = Temp_K - 273.15 
    
    ### Open specific humidity data and select winter months (DJF) ###
    huss_fpath = f'/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/output/netcdf/macav2metdata_GSLBIP_{model_name}_{emission_scenario}_huss.nc'
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
    
    from pathlib import Path

    fpath = Path(f'/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/press/'
                 f'macav2metdata_GSLBIP_{model_name}_{emission_scenario}_press.nc')

    if fpath.exists():
        print('File exists!')
        print(f'Pressure data found in: {fpath}')
        open_P = xr.open_dataset(fpath) 
        P = open_P['press']
    else:
        ### calculate pressure based on elevation data ###
        Ps = 1013.25 # sea level pressure (hPa)
        H = 7600 # scale height (m)
        P = Ps * np.exp(-elevation / H)
        # P = P.load()
        
        P.name = 'press'
        P.attrs['units'] = 'hPa'
        P.to_netcdf(fpath)
        
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
                
                
    wbt.name = 'wbt'
    wbt.attrs['units'] = 'Degrees C'
    wbt.to_netcdf(f'/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/wbt/macav2metdata_GSLBIP_{model_name}_{emission_scenario}_wbt.nc')
    
    return P, wbt

P, wbt = wbt_press_calc('ACCESS-CM2', 'ssp585')

