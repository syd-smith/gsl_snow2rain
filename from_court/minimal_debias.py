#%%
"""
Author: Court Strong
Date Created: October 13, 2026
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import xarray as xr

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from WRF.debias.plots.validation import running_window_slice, loc_sel

def make_series(seed, n, bias=0.0, trend=0.25 / 30, phi=0.7, sigma=3.0, base=295.0):
    """Red noise (AR(1)) with linear trend; trend is K per sample (0.25 K/yr over 30 days/yr)."""
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    noise[0] = rng.normal(0, sigma)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.normal(0, sigma * np.sqrt(1 - phi**2))
    return base + bias + trend * np.arange(n) + noise

def compute_ecdf(data):
    # 1. Sort the vector
    x_sorted = np.sort(data)
    n = len(x_sorted)
    
    # 2. Assign values from 1/n to n/n (1.0)
    y_vals = np.arange(1, n + 1) / n
    
    return x_sorted, y_vals

def detrend(data):
    """Remove the linear trend, keeping the original mean. Returns (detrended, params)."""
    t = np.arange(len(data))
    slope = np.polyfit(t, data, 1)[0]
    t_mean = t.mean()
    return data - slope * (t - t_mean), (slope, t_mean)

def retrend(data, params):
    """Add back the trend removed by detrend()."""
    slope, t_mean = params
    t = np.arange(len(data))
    return data + slope * (t - t_mean)

def make_bias_function(mod_hist, obs):
    """Return bias(p) = F_mod_hist^-1(p) - F_obs^-1(p), and F_mod_hist(x) as callables."""
    mx, mp = compute_ecdf(mod_hist)
    ox, op = compute_ecdf(obs)
    p_grid = np.linspace(0, 1, 1001)
    bias_grid = np.interp(p_grid, mp, mx) - np.interp(p_grid, op, ox)
    bias = lambda p: np.interp(p, p_grid, bias_grid)
    cdf = lambda x: np.interp(x, mx, mp)
    return bias, cdf

def quantile_map(data, bias, cdf):
    """Subtract the quantile-dependent bias, evaluated at each value's historical-model CDF."""
    return data - bias(cdf(data))

obs = make_series(seed=12, n=35 * 30)
mod = make_series(seed=3, n=120 * 30, bias=5.0)

lat = 40.788
lon = -111.978
date = '1985-06-15'
var = 'tmmx'
obs_path =  glob.glob(str(parent_dir / 'WRF' / 'debias' / 'gridMET' / f'*tmmx*.nc'))
obs_all_syd = xr.open_dataset(obs_path[0], decode_times = True)
obs_loc = loc_sel(obs_all_syd, var, lat=lat, lon=lon)
obs_syd = running_window_slice(obs_loc, date)


mod_path =  glob.glob(str(parent_dir / 'WRF' / 'debias' / 'daily' / f'*tmmx_safe*.nc'))
mod_all_syd = xr.open_dataset(mod_path[0], decode_times = True)
mod_loc = loc_sel(mod_all_syd, var, lat=lat, lon=lon)
mod_syd = running_window_slice(mod_loc, date)

obs_dt, obs_params = detrend(obs_syd)
mod_dt, mod_params = detrend(mod_syd)

assert np.allclose(retrend(obs_dt, obs_params), obs_syd) and np.allclose(retrend(mod_dt, mod_params), mod_syd)

bias, cdf = make_bias_function(mod_dt[:35 * 30], obs_dt)
mod_bc_dt = quantile_map(mod_dt, bias, cdf)
# Pivot the trend on the historical period so the corrected history matches obs
mod_bc = retrend(mod_bc_dt, (mod_params[0], np.arange(35 * 30).mean()))
print(f"Obs mean {obs.mean():.2f} K; model hist mean raw {mod[:1050].mean():.2f} K, "
      f"debiased {mod_bc[:1050].mean():.2f} K")

fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
for ax, o, m, title in [(axes[0], obs_syd, mod_syd, "Raw"), (axes[1], obs_dt, mod_dt, "Detrended"),
                         (axes[2], obs_dt, mod_bc_dt, "Debiased (detrended)"),
                         (axes[3], obs_syd, mod_bc, "Retrended (after debiasing)")]:
    ax.plot(m, lw=0.5, color="tab:orange", label="Model")
    ax.plot(o, lw=0.5, color="tab:blue", label="Obs")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(title)
    ax.legend()
axes[3].set_xlabel("Day index (June days concatenated)")
fig.tight_layout()
fig.savefig(f"red_noise_series_slc_airport_{date}.png", dpi=150)

# mod_hist = mod[:35 * 30]
# mod_fut = mod[35 * 30:]

# plt.figure(figsize=(6, 4))
# for x, c, lw, lab in [(obs, "k", 3, "Obs 1980-2014"),
#                       (mod_hist, "tab:orange", 1.5, "Model 1980-2014"),
#                       (mod_fut, "tab:red", 1.5, "Model 2015-2099"),
#                       (mod_bc[:35 * 30], "m", 1.5, "Model 1980-2014 debiased"),
#                       (mod_bc[35 * 30:], "tab:purple", 1.5, "Model 2015-2099 debiased")]:
#     plt.plot(*compute_ecdf(x), color=c, lw=lw, label=lab)
# plt.xlabel("Temperature (K)")
# plt.ylabel("CDF")
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"cdfs_slc_airport_{date}.png", dpi=150)
# plt.show()

# %%
