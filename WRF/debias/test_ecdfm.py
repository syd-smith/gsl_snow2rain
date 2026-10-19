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
from plots.validation import running_window_slice, loc_sel

# ======================
# - Create Dummy Data - 
# ======================

def ds_dummy(start_day = '1985-01-01', stop_day = '1985-12-31', var = 'tmmx', trend = True):
    # Define dimension values
    time_vals = pd.date_range(start = start_day, end = stop_day, freq = 'D')

    # Create location dims and broadcast over time dim
    lats = np.array([[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]])
    lat_vals = np.broadcast_to(lats, (len(time_vals), *lats.shape))
    lons = np.array([[-105, -106, -107, -108, -109], [-110, -111, -112, -113, -114], [-115, -116, -117, -118, -119], [-120, -121, -122, -123, -124]])
    lon_vals = np.broadcast_to(lons, (len(time_vals), *lons.shape))

    # Ensure lat and lon vals have same shape
    assert lat_vals.shape == lon_vals.shape, 'Lat and lon values must have same shape.'

    if trend:
        # Create simple array of data with linear trend
        data = np.full((4, 5), 275)
        trend_1d = np.arange(len(time_vals)) * 0.2 # Data has a slope of 0.2K per day
        var_data = data + trend_1d[:, None, None]

    else: 
        # Create simple array of data with linear trend
        data = np.full((4, 5), 275)
        var_data = np.broadcast_to(data, (len(time_vals), *data.shape))

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

def ds_simplest_case():
    # Define dimension values
    time_vals = pd.date_range(start = '1985-01-01', end = '1985-01-05', freq = 'D')

    # Create location dims and broadcast over time dim
    lats = np.array([[40], [45]])
    lat_vals = np.broadcast_to(lats, (len(time_vals), *lats.shape))
    lons = np.array([[-105], [-110]])
    lon_vals = np.broadcast_to(lons, (len(time_vals), *lons.shape))

    # Ensure lat and lon vals have same shape
    assert lat_vals.shape == lon_vals.shape, 'Lat and lon values must have same shape.'

    # Create data
    data = np.full((2, 2), [1, 2])
    var_data = np.broadcast_to(data, (len(time_vals), *data.shape))

    # Ensure variable data is same shape as location data
    assert var_data.shape == lon_vals.shape

    # Recognize that dims should be 3D
    dims_3d = ('time', 'south_north', 'west_east')

    # Create new dataset to save data to
    clean_ds = xr.Dataset(
        {
            'tmmx': (['time', 'lat', 'lon'], var_data)
        },
        coords = {
            'time': ('time', time_vals),
            'lat': (dims_3d, lat_vals),
            'lon': (dims_3d, lon_vals)
        }
    )

    return clean_ds

def ds_uniform(seed):
    # Initialize number generator
    rng = np.random.default_rng(seed = seed)

    # Define dimension values
    time_vals = pd.date_range(start = '1985-01-01', end = '1985-01-31', freq = 'D')

    # Create location dims and broadcast over time dim
    lats = np.array([[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]])
    lat_vals = np.broadcast_to(lats, (len(time_vals), *lats.shape))
    lons = np.array([[-105, -106, -107, -108, -109], [-110, -111, -112, -113, -114], [-115, -116, -117, -118, -119], [-120, -121, -122, -123, -124]])
    lon_vals = np.broadcast_to(lons, (len(time_vals), *lons.shape))

# ==================
# - Test Fucntions - 
# ==================

def test_trend():

    # Call dumy dataset
    data = ds_dummy()

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

def test_running_window():
    # Call dummy dataset without a trend
    ds = ds_dummy(trend = False)

    # Set date of interest
    date = '1985-03-05'

    # Call dummy dataset with constrained date range
    ds_sliced_ctrl = ds_dummy(start_day = '1985-02-18', stop_day = '1985-03-20', trend = False)

    # Slice data using running_window_slice
    ds_sliced_test = running_window_slice(ds, date)

    xr.testing.assert_allclose(ds_sliced_ctrl, ds_sliced_test, rtol = 1e-5, atol = 1e-5)



def test_doy_mask():

def test_step_func():

def test_vect_ECDF():

def test_vect_ECDF_inverse():



# %%
