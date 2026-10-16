#%%
"""
Author: Sydney Smith
Date Created: October 15, 2026
"""

import glob
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import xarray as xr

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(current_dir))

from validation import running_window_slice, loc_sel

var = 'tmmx'
date = '1985-10-15'
lat = 40.788
lon = -111.978


obs_all_syd = xr.open_dataset('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/gridMET/gridMET_GSLBIP_tmmx.nc', decode_times = True)
obs_loc = loc_sel(obs_all_syd, var, lat=lat, lon=lon)
obs_syd = running_window_slice(obs_loc, date)

mod_all_syd = xr.open_dataset('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/daily/wrf_raw_GSLBIP_multimodel_ssp245_tmmx_safe.nc', decode_times = True)
mod_loc = loc_sel(mod_all_syd, var, lat=lat, lon=lon)
mod_syd = running_window_slice(mod_loc, date)

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

fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
for ax, o, m, title in [(axes[0], obs_syd, mod_syd, "Raw"), (axes[1], obs_dt, mod_dt, "Detrended"),
                         (axes[2], obs_dt, mod_bc_dt, "Debiased (detrended)"),
                         (axes[3], obs_syd, mod_bc_rt, "Retrended (after debiasing)")]:
    ax.plot(m, lw=0.5, color="tab:orange", label="Model")
    ax.plot(o, lw=0.5, color="tab:blue", label="Obs")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(title)
    ax.legend()
axes[3].set_xlabel("Day index (June days concatenated)")
fig.tight_layout()
fig.savefig(f"MY_red_noise_series_slc_airport_{date}.png", dpi=150)
# %%
