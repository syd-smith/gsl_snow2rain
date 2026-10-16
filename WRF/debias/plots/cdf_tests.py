#%%
"""
Author: Sydney Smith & Court Strong
Date Created: October 16, 2026
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import xarray as xr

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(current_dir))

from validation import running_window_slice, loc_sel

var = 'tmmx'
date = '1985-04-15'
lat = 40.788
lon = -111.978


obs_all_syd = xr.open_dataset('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/gridMET/gridMET_GSLBIP_tmmx.nc', decode_times = True)
obs_loc = loc_sel(obs_all_syd, var, lat=lat, lon=lon)
obs = running_window_slice(obs_loc, date)

mod_all_syd = xr.open_dataset('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/daily/wrf_raw_GSLBIP_multimodel_ssp245_tmmx_safe.nc', decode_times = True)
mod_loc = loc_sel(mod_all_syd, var, lat=lat, lon=lon)
mod = running_window_slice(mod_loc, date)

mod_hist = mod.sel(time = slice('1985-01-01', '2014-12-31'))
mod_fut = mod.sel(time = slice('2015-01-01', '2099-12-31'))

wrfout = '/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/wrfout/'

obs_dt_open = xr.open_dataset(wrfout+'detrended_obs_tmmx.nc', decode_times = True)
obs_dt_loc = loc_sel(obs_dt_open, var, lat=lat, lon=lon)
obs_dt = running_window_slice(obs_dt_loc, date)

mod_dt_open = xr.open_dataset(wrfout+'detrended_raw_tmmx.nc', decode_times = True)
mod_dt_loc = loc_sel(mod_dt_open, var, lat=lat, lon=lon)
mod_dt = running_window_slice(mod_dt_loc, date)

debiased = xr.open_dataset(wrfout+'debiased_tmmx.nc', decode_times = True)
debiased_loc = loc_sel(debiased, var, lat=lat, lon=lon)
mod_bc_dt = running_window_slice(debiased_loc, date)

retrended = xr.open_dataset(wrfout+'wrfout_GSLBIP_multimodel_ssp245_tmmx_test.nc', decode_times = True)
rt_loc = loc_sel(retrended, var, lat=lat, lon=lon)
mod_bc_rt = running_window_slice(rt_loc, date)

mod_bc_rt_hist = mod_bc_rt.sel(time = slice('1985-01-01', '2014-12-31'))
mod_bc_rt_fut = mod_bc_rt.sel(time = slice('2015-01-01', '2099-12-31'))

def compute_ecdf(data):
    # 1. Sort the vector
    x_sorted = np.sort(data)
    n = len(x_sorted)
    
    # 2. Assign values from 1/n to n/n (1.0)
    y_vals = np.arange(1, n + 1) / n
    
    return x_sorted, y_vals

plt.figure(figsize=(6, 4))
for x, c, lw, lab in [(obs, "k", 3, "Obs 1980-2014"),
                      (mod_hist, "tab:orange", 1.5, "Model 1980-2014"),
                      (mod_fut, "tab:red", 1.5, "Model 2015-2099"),
                      (mod_bc_rt_hist, "m", 1.5, "Model 1980-2014 debiased"),
                      (mod_bc_rt_fut, "tab:purple", 1.5, "Model 2015-2099 debiased")]:
    plt.plot(*compute_ecdf(x), color=c, lw=lw, label=lab)
plt.xlabel("Temperature (K)")
plt.ylabel("CDF")
plt.legend()
plt.tight_layout()
plt.savefig(f"cdfs_slc_airport_{date}.png", dpi=150)
plt.show()
# %%
