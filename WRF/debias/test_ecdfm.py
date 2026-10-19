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
#%%
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

def ds_dummy(start_day = '1985-01-01', stop_day = '1985-12-31', var = 'tmmx', trend = 0):
    # Define dimension values
    time_vals = pd.date_range(start = start_day, end = stop_day, freq = 'D')

    # Create location dims and broadcast over time dim
    lats = np.array([[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]])
    lat_vals = np.broadcast_to(lats, (len(time_vals), *lats.shape))
    lons = np.array([[-105, -106, -107, -108, -109], [-110, -111, -112, -113, -114], [-115, -116, -117, -118, -119], [-120, -121, -122, -123, -124]])
    lon_vals = np.broadcast_to(lons, (len(time_vals), *lons.shape))

    # Ensure lat and lon vals have same shape
    assert lat_vals.shape == lon_vals.shape, 'Lat and lon values must have same shape.'

    # Create simple array of data with linear trend
    data = np.full((4, 5), 275)

    # Default set to have no trend
    if trend == 0:
        var_data = np.broadcast_to(data, (len(time_vals), *data.shape))

    else:
        # Handle trends passed as strings
        if isinstance(trend, str):
            op = trend[0]
            val = float(trend[1:])
        else: 
            op = '+'
            val = float(trend)

        # Multipicative trend
        if op == '*':
            trend_1d = np.arange(len(time_vals)) * val # Data has a slope of 0.2K per day
        # Additive trend
        else:
            trend_1d = np.arange(len(time_vals)) + val # Data has a slope of 0.2K per day

        # Cast trend to 3D shape
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

# def ds_simplest_case():
# Define dimension values
time_vals = pd.date_range(start = '1985-01-01', end = '1985-01-05', freq = 'D')

# Create location dims and broadcast over time dim
lats = np.array([[40, 42], [45, 47]])
lat_vals = np.broadcast_to(lats, (len(time_vals), *lats.shape))
lons = np.array([[-105, -107], [-110, -112]])
lon_vals = np.broadcast_to(lons, (len(time_vals), *lons.shape))

# Ensure lat and lon vals have same shape
assert lat_vals.shape == lon_vals.shape, 'Lat and lon values must have same shape.'

# Create data
data = np.linspace(1, len(time_vals), len(time_vals))
var_data = np.broadcast_to(data[:, None, None], lon_vals.shape)

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

    # return clean_ds

def ds_dist(seed, dist):
    # Initialize number generator
    rng = np.random.default_rng(seed = seed)

    # Define dimension values
    time_vals = pd.date_range(start = '1985-01-01', end = '1985-01-31', freq = 'D')

    # Create location dims and broadcast over time dim
    lats = np.array([[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]])
    lat_vals = np.broadcast_to(lats, (len(time_vals), *lats.shape))
    lons = np.array([[-105, -106, -107, -108, -109], [-110, -111, -112, -113, -114], [-115, -116, -117, -118, -119], [-120, -121, -122, -123, -124]])
    lon_vals = np.broadcast_to(lons, (len(time_vals), *lons.shape))

    # Ensure lat and lon vals have same shape
    assert lat_vals.shape == lon_vals.shape, 'Lat and lon values must have same shape.'

    # Distribution should be a 1D array with length of time values
    n = len(time_vals)

    if dist == 'uniform':
        data = rng.uniform(low = 275, high = 300, size = n)

    elif dist == 'normal':
        # loc: center of bell curve
        # scale: width of distribution
        # size: number of samples to generate
        data = rng.normal(loc = 287.5, scale = 3, size = n)

    elif dist == 'gamma':
        # Gamma distributions have a baseline of zero so generate temp in excess of 275
        temp_excess = rng.gamma(3, 4, size = n) # mean excess is shape * scale = 20
        data = 275 + temp_excess

    elif dist == 'beta':
        # Beta distribution strictly calculates values between 0 and 1
        # If alpha>beta then skewed to right
        min_val = 275
        max_val = 300
        beta_vals = rng.beta(a = 4, b = 3, size = n)

        # Scale beta_vals to the actual range
        data = min_val + (beta_vals * (max_val - min_val))

    # Broadcast data to be the same 1D time series at each location
    var_data = np.broadcast_to(data[:, None, None], lon_vals.shape)

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

    # Ensure data values are repeated with the same trend over time at each location
    assert (clean_ds['tmmx'].isel(lat = 3, lon = 4).values == clean_ds['tmmx'].isel(lat = 0, lon = 0).values).all(), 'Data points must be the same at each location.'

    return clean_ds

#%%
# ==================
# - Test Fucntions - 
# ==================

def test_trend():

    # Call dumy dataset with a trend of 0.2
    data = ds_dummy(trend = '*0.2')

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
    ds = ds_dummy()

    # Set date of interest
    date = '1985-03-05'

    # Call dummy dataset with constrained date range and no trend
    ds_sliced_ctrl = ds_dummy(start_day = '1985-02-18', stop_day = '1985-03-20')

    # Slice data using running_window_slice
    ds_sliced_test = running_window_slice(ds, date)

    xr.testing.assert_allclose(ds_sliced_ctrl, ds_sliced_test, rtol = 1e-5, atol = 1e-5)

def test_doy_mask():
    # Call dummy dataset without a trend
    ds = ds_dummy()

    # Set date of interest
    date = '1985-03-05'

    # Call dummy dataset with constrained date range
    ds_sliced_ctrl = ds_dummy(start_day = '1985-03-05', stop_day = '1985-03-05')

    # Slice data using running_window_slice
    ds_sliced_test = doy_mask(ds, date)

    xr.testing.assert_allclose(ds_sliced_ctrl, ds_sliced_test, rtol = 1e-5, atol = 1e-5)

class TestStepFunc3D:

    @pytest.fixture
    def sample_data(self):
        """
        Generate sample 3D sample data (time, lat, lon).
        """
        x = np.array([
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
            [[3, 3], [3, 3]],
            [[4, 4], [4, 4]],
            [[5, 5], [5, 5]]
        ])
        y = np.array([
            [[0.2, 0.2], [0.2, 0.2]],
            [[0.4, 0.4], [0.4, 0.4]],
            [[0.6, 0.6], [0.6, 0.6]],
            [[0.8, 0.8], [0.8, 0.8]],
            [[1, 1], [1, 1]]
        ])
        return x, y

    @pytest.fixture
    def step_func_right(self, sample_data):
        """
        Instantiates StepFunc3D instance using right side.
        """
        x, y = sample_data
        return StepFunc3D(x, y, side = 'right', ival = 0)

    # --- Test Cases ___

    def test_init(self, sample_data):
        # Test the instance initializes properly
        x, y = sample_data
        sf = StepFunc3D(x, y, side = 'right')

        assert sf.n_steps == 5
        assert sf.side == 'right'
        assert np.array_equal(sf.x, x)

    def test_side_error(self, sample_data):
        # Test that code errors out if improper side string is provided
        x, y, = sample_data
        with pytest.raises(ValueError, match = 'Side can take the values "right" or "left."'):
            StepFunc3D(x, y, side = 'middle') # Intentionally pass bad string

    def test_mismatch_shapes(self):
        # Test that different x and y shapes triggers error
        # Create x and y with a mismatched shape
        x = np.ones((5, 2, 2))
        y = np.ones((5, 1, 2))
        with pytest.raises(ValueError, match = 'x and y must have the same shape'):
            StepFunc3D(x, y)

    def test_call(self, step_func_right):
        # Test that output is callable
        # Lookup the values passed in a 2D array
        lookup = np.array([
            [2, 4],
            [0.5, 5] # Values below 1 should return ival = 0
        ])

        result = step_func_right(lookup)

        # Define expected output
        expected = np.array([[
            [0.4, 0.8],
            [0, 1]
        ]])

        np.testing.assert_array_equal(result, expected)

class Testvectorized_ECDF:

    @pytest.fixture
    def sample_data(self):
        """
        Generate sample 3D sample data (time, lat, lon).
        """
        x = np.array([
            [[1, 1], [1, 1]],
            [[3, 3], [3, 3]],
            [[2, 2], [2, 2]],
            [[4, 4], [4, 4]],
            [[5, 5], [5, 5]]
        ])
        y = np.array([
            [[0.2, 0.2], [0.2, 0.2]],
            [[0.4, 0.4], [0.4, 0.4]],
            [[0.6, 0.6], [0.6, 0.6]],
            [[0.8, 0.8], [0.8, 0.8]],
            [[1, 1], [1, 1]]
        ])
        return x, y

    @pytest.fixture
    def data_right(self, sample_data):
        """
        Instantiates vectorized_ECDF instance using right side.
        """
        x, y = sample_data
        return vectorized_ECDF(x)

    # --- Test Cases ---

    def test_init(self, sample_data):
        # Test the instance initializes properly
        x, y = sample_data
        ecdf = vectorized_ECDF(x)

        # Sort x data mannually (this is the expected output)
        x_sorted = np.array([
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
            [[3, 3], [3, 3]],
            [[4, 4], [4, 4]],
            [[5, 5], [5, 5]]
        ])

        # Ensure results match expected
        assert ecdf.n_steps == 5
        assert ecdf.side == 'right'
        assert np.array_equal(ecdf.x, x_sorted)
        assert np.array_equal(ecdf.y, y)

    def test_inverse(self, sample_data):
        # Call sample data
        x, y = sample_data

        # Results from function in question
        ecdf = vectorized_ECDF(x)
        ecdf_inv = ecdf.inverse(y)

        # Sort x data mannually to verify outputs match
        x_sorted = np.array([
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
            [[3, 3], [3, 3]],
            [[4, 4], [4, 4]],
            [[5, 5], [5, 5]]
        ])

        # Ensure they match
        assert np.array_equal(ecdf_inv, x_sorted)

