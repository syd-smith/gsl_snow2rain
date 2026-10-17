#%%
"""
Author: Sydney Smith
Date Created: October 16, 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import sys
import xarray as xr

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from ecdfm_man import StepFunc3D, vectorized_ECDF, doy_mask, vect_app, detrend_ds, retrend_ds

# ======================
# - Create Dummy Data - 
# ======================

def dummy_trend_ds():
    var = 'tmmx'

    # Define dimension values
    time_vals = pd.date_range(start = '1985-01-01', end = '1985-12-31', freq = 'D')

    # Create location dims and broadcast over time dim
    lats = np.array([[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]])
    lat_vals = np.broadcast_to(lats, (len(time_vals), *lats.shape))
    lons = np.array([[-105, -106, -107, -108, -109], [-110, -111, -112, -113, -114], [-115, -116, -117, -118, -119], [-120, -121, -122, -123, -124]])
    lon_vals = np.broadcast_to(lons, (len(time_vals), *lons.shape))

    # Ensure lat and lon vals have same shape
    assert lat_vals.shape == lon_vals.shape, 'Lat and lon values must have same shape.'

    # Create simple array of data with linear trend
    data = np.full((4, 5), 275)
    trend_1d = np.arange(len(time_vals)) * 0.2 # Data has a slope of 0.2K per day
    var_data = data + trend_1d[:, None, None]

    # Ensure variable data is same shape as location data
    assert var_data.shape == lon_vals.shape

    # Recognize that dims should be 3D
    dims_3d = ('time', 'south_north', 'west_east')

    # Create new dataset to save data to
    clean_ds = xr.Dataset(
        {
            var: (['time', 'lat', 'lon'], var_data)
        },
        coords = {
            'time': ('time', time_vals),
            'lat': (dims_3d, lat_vals),
            'lon': (dims_3d, lon_vals)
        }
    )

    # Ensure data values are repeated with the same trend over time at each location
    assert (clean_ds[var].isel(lat = 3, lon = 4).values == clean_ds[var].isel(lat = 0, lon = 0).values).all(), 'Data points must be the same at each location.'

    return clean_ds

# ==================
# - Test Fucntions - 
# ==================

def test_trend():

    # Call dumy dataset
    data = dummy_trend_ds()

    # Pull the time values from the dataset
    time_idx = np.arange(len(data['time'].values))
    # Use numpy to find y=mx+b
    slope, intercept = np.polyfit(time_idx, data['tmmx'].isel(lat = 0, lon = 0).values, 1)

    # Ensure that what numpy calculated is close to what was expected
    assert np.isclose(slope, 0.2)
    assert np.isclose(intercept, data['tmmx'].values[0][0][0])

    # Calculate the trend
    trend = (slope * time_idx) + intercept
    trend_3D = np.broadcast_to(trend[:, None, None], data['tmmx'].shape)

    # Detrend the data
    detrended_data = data['tmmx'] - trend_3D

    # Calculate the data's mean and broadcast it to a 3D array
    data_mean = np.mean(data['tmmx'].isel(lat = 0, lon = 0).values)
    detrended_with_mean = detrended_data + data_mean

    # Call function that is being tested
    func_detrended, func_trend, func_avg = detrend_ds(data, 'tmmx')

    # Make assertions based on manual detrending
    np.testing.assert_allclose(func_detrended['tmmx'].values, detrended_with_mean.values, rtol = 1e-5, atol = 1e-5)
    np.testing.assert_allclose(func_trend, trend_3D, rtol = 1e-5, atol = 1e-5)
    assert np.isclose(func_avg.isel(lat = 0, lon = 0), data_mean)


# %%
