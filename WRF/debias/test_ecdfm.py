#%%
"""
Author: Sydney Smith
Date Created: October 16, 2026
"""

import glob
import numpy as np
import os
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

from ecdfm_man import StepFunc3D, vectorized_ECDF, ECDFM, doy_mask, vect_app, detrend_ds, retrend_ds
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

def ds_simplest_case():
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

    return clean_ds

def ds_dist(seed, dist, min_val = 275, max_val = 310, mean_val = 293):
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
        data = rng.uniform(low = min_val, high = max_val, size = n)

    elif dist == 'normal':
        # loc: center of bell curve
        # scale: width of distribution
        # size: number of samples to generate
        data = rng.normal(loc = mean_val, scale = 3, size = n)

    elif dist == 'gamma':
        # Gamma distributions have a baseline of zero so generate temp in excess of 275
        temp_excess = rng.gamma(3, 4, size = n) # mean excess is shape * scale = 20
        data = min_val + temp_excess

    elif dist == 'beta':
        # Beta distribution strictly calculates values between 0 and 1
        # If alpha>beta then skewed to right
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

# ====================
# - Pytest Fucntions - 
# ====================

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

# ========================
# - Other Test Fucntions - 
# ========================

def nan_test(var):
    """
    Returns location of Nans in debiased data.
    """
    fpath = glob.glob(str(current_dir / 'wrfout' / f'*{var}*.nc'))
    debiased = xr.open_dataset(fpath[0], decode_times = True)

    nan_data = debiased[var].where(debiased[var].isnull(), drop = True)

    return nan_data

def np_nan_loc(arr):
    """
    Find locations of NaNs in np.array.
    """
    return np.argwhere(np.isnan(arr))

def debias_hist_only(var, vectorized = True, save = True):
    """
    Debias historical period in isolation.
    """

    # Open datasets
    obs_path = glob.glob(str(current_dir / 'gridMET' / f'*{var}*.nc'))
    obs = xr.open_dataset(obs_path[0], decode_times = True)

    # Slice model data down to only historical period
    mod_path = glob.glob(str(current_dir / 'daily' / f'*{var}*.nc'))
    mod = xr.open_dataset(mod_path[0], decode_times = True).sel(time = slice('1985-01-01', '2014-12-31'))

    # Subtract the trend from the dataset
    detrended_obs, obs_trend, obs_mean = detrend_ds(obs, var)
    detrended_mod, mod_trend, mod_mean = detrend_ds(mod, var)

    # Set location of interest as SLC airport
    lat = 40.788
    lon = -111.978

    # Slice model data down to historical period
    hist = detrended_mod.sel(time = slice('1985-01-01', '2014-12-31'))

    # Pass datasets to debiaser
    debiased_mod = ECDFM(detrended_obs, hist, detrended_mod, var, lat, lon, vectorized = vectorized)

    # Put trend back in data
    retrended_mod = retrend_ds(debiased_mod, var, mod_trend, mod_mean)

    if save:
        # Create output directory to store new cleaned files
        output_dir = current_dir / 'wrfout_hist'
        os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

        # Generate save name for debiased data
        detrended_obs_path = os.path.join(output_dir, f'detrended_obs_{var}.nc')
        detrended_mod_path = os.path.join(output_dir, f'detrended_mod_{var}.nc')
        debiased_mod_path = os.path.join(output_dir, f'debiased_{var}.nc')
        out_path = os.path.join(output_dir, f'wrfout_GSLBIP_multimodel_ssp245_{var}_test.nc')

        # Save data to netCDF file
        detrended_obs.to_netcdf(detrended_obs_path)
        detrended_mod.to_netcdf(detrended_mod_path)
        debiased_mod.to_netcdf(debiased_mod_path)
        retrended_mod.to_netcdf(out_path)

    # Close out of data
    del obs_trend
    del obs_mean
    del mod_trend
    del mod_mean
    detrended_obs.close()
    detrended_mod.close()
    debiased_mod.close()
    retrended_mod.close()

def debias_dist(var, obs_dist, mod_dist, save_name, vectorized = True):
    """
    Debias historical period in isolation.
    """

    # Open datasets
    obs = ds_dist(seed = 20, dist = 'uniform')
    mod = ds_dist(seed = 25, dist = 'uniform', min_val = 280, max_val = 315)

    # Subtract the trend from the dataset
    detrended_obs, obs_trend, obs_mean = detrend_ds(obs, var)
    detrended_mod, mod_trend, mod_mean = detrend_ds(mod, var)

    # Set location of interest as SLC airport
    lat = 40.788
    lon = -111.978

    # Pass datasets to debiaser
    debiased_mod = ECDFM(detrended_obs, detrended_mod, var, lat, lon, vectorized = vectorized)

    # Put trend back in data
    retrended_mod = retrend_ds(debiased_mod, var, mod_trend, mod_mean)

    # Create output directory to store new cleaned files
    output_dir = current_dir / save_name
    os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

    # Generate save name for debiased data
    detrended_obs_path = os.path.join(output_dir, f'detrended_obs_{var}.nc')
    detrended_mod_path = os.path.join(output_dir, f'detrended_mod_{var}.nc')
    debiased_mod_path = os.path.join(output_dir, f'debiased_{var}.nc')
    out_path = os.path.join(output_dir, f'wrfout_GSLBIP_multimodel_ssp245_{var}_test.nc')

    # Save data to netCDF file
    detrended_obs.to_netcdf(detrended_obs_path)
    detrended_mod.to_netcdf(detrended_mod_path)
    debiased_mod.to_netcdf(debiased_mod_path)
    retrended_mod.to_netcdf(out_path)

def fut_hist_isolated(var, vectorized = True, save = True):
    """
    Debias future period in isolation.
    """

    # Open datasets
    obs_path = glob.glob(str(current_dir / 'gridMET' / f'*{var}*.nc'))
    obs = xr.open_dataset(obs_path[0], decode_times = True)

    # Slice model data down to only historical period
    mod_path = glob.glob(str(current_dir / 'daily' / f'*{var}*.nc'))
    mod = xr.open_dataset(mod_path[0], decode_times = True).sel(time = slice('1985-01-01', '2014-12-31'))

    # Separate into future and historical
    hist = mod.sel(time = slice('1985-01-01', '2014-12-31'))
    fut = mod.sel(time = slice('2015-01-01', '2099-12-31'))

    # Subtract the trend from the dataset
    detrended_obs, obs_trend, obs_mean = detrend_ds(obs, var)
    detrended_hist, hist_trend, hist_mean = detrend_ds(hist, var)
    detrended_fut, fut_trend, fut_mean = detrend_ds(fut, var)

    # Set location of interest as SLC airport
    lat = 40.788
    lon = -111.978

    # Pass datasets to debiaser
    debiased_hist = ECDFM(detrended_obs, detrended_hist, detrended_hist, var, lat, lon, vectorized = vectorized)
    debiased_fut = ECDFM(detrended_obs, detrended_hist, detrended_fut, var, lat, lon, vectorized = vectorized)

    # Put trend back in data
    retrended_hist = retrend_ds(debiased_hist, var, hist_trend, hist_mean)
    retrended_fut = retrend_ds(debiased_fut, var, fut_trend, fut_mean)

    # Slice dataset down to correct time periods
    retrended_hist = retrended_hist.sel(time = slice('1985-01-01', '2014-12-31'))
    retrended_fut = retrended_fut.sel(time = slice('2015-01-01', '2099-12-31'))

    if save:
        # Create output directory to store new cleaned files
        output_dir = current_dir / 'wrfout_hist_fut_sep'
        os.makedirs(output_dir, exist_ok = True) # Don't make if it already exists

        # Generate save name for debiased data
        detrended_obs_path = os.path.join(output_dir, f'detrended_obs_{var}.nc')
        detrended_hist_path = os.path.join(output_dir, f'detrended_hist_{var}.nc')
        detrended_fut_path = os.path.join(output_dir, f'detrended_fut_{var}.nc')
        debiased_hist_path = os.path.join(output_dir, f'debiased_hist_{var}.nc')
        debiased_fut_path = os.path.join(output_dir, f'debiased_fut_{var}.nc')
        retrended_hist_path = os.path.join(output_dir, f'retrended_hist_{var}.nc')
        retrended_fut_path = os.path.join(output_dir, f'retrended_fut_{var}.nc')

        # Save data to netCDF file
        detrended_obs.to_netcdf(detrended_obs_path)
        detrended_hist.to_netcdf(detrended_hist_path)
        detrended_fut.to_netcdf(detrended_fut_path)
        debiased_hist.to_netcdf(debiased_hist_path)
        debiased_fut.to_netcdf(debiased_fut_path)
        retrended_hist.to_netcdf(retrended_hist_path)
        retrended_fut.to_netcdf(retrended_fut_path)

fut_hist_isolated('tmmx')

# debias_hist_only('tmmx', current_dir)