# %%
"""
Author: Sydney Smith
Date: June 16, 2026
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from wrf import (ALL_TIMES, getvar)
import xarray as xr

SWE_output = []
# location of wrf files
fpath = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/'

for domain in ['d01', 'd02', 'd03']:
    # pull all April first data for d02 to take average of SWE
    wrf_files = sorted(glob.glob(f'{fpath}/wrfout_{domain}*04-01_*'))
    # open all collected files
    open_files = [Dataset(file) for file in wrf_files]
    SWE = getvar(open_files, 'SNOW', timeidx = ALL_TIMES)

    # take average April 1st SWE across historical period
    avg_SWE = SWE.mean(dim = 'Time')
    # add to output list
    SWE_output.append(avg_SWE)

# %%
# plot average SWE
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 6), subplot_kw = {'projection': ccrs.PlateCarree()})
flat_axes = ax.flatten()

for idx, ax in enumerate(flat_axes):
    # plot data
    contour = ax.contourf(SWE_output[idx]['XLONG'], SWE_output[idx]['XLAT'], SWE_output[idx], cmap = 'viridis')
    
    # set map extent
    lons = SWE_output[idx]['XLONG']
    lats = SWE_output[idx]['XLAT']
    
    # Set the extent with a buffer
    extent = [float(lons[0][0]), float(lons[0][-1]), float(lats[2][0]), float(lats[-1][0])]
    ax.set_extent(extent, crs = ccrs.PlateCarree())

    # add map features to orient within the map
    states = cfeature.NaturalEarthFeature(category = 'cultural', name = 'admin_1_states_provinces_lines', scale = '50m', facecolor = 'none', edgecolor = 'k', zorder = 4)
    ax.add_feature(states, linewidth = 1.3)
    ax.add_feature(cfeature.LAKES, zorder = 2, linewidth = 1)
    ax.add_feature(cfeature.RIVERS, linewidth = 1, zorder = 1)
    ax.coastlines(resolution = '50m', color = 'black', linewidth = 1)

# add colorbar for entire figure
fig.colorbar(contour, cax = fig.add_axes([0.15, 0.24, 0.7, 0.04]), label = 'Average SWE (kg m-2)', orientation = 'horizontal')

# display plot
plt.show()




# %%