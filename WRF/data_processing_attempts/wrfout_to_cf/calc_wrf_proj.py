"""
Created: February 2024
Author: Savanna Wolvin, s.wolvin@utah.edu

Edited: July 15, 2026
By: Sydney Smith

map projections based on 
https://fabienmaussion.info/2018/01/06/wrf-projection/
"""
#%% Imports
import numpy as np
import netCDF4 as nc
from wrf import get_cartopy
import os
import pyproj


#%% Variable Preset
YEAR = 1985

#file location
file_path = '/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/'
NEST = 'd03'

#%% Load data
files = os.listdir(file_path)
ncfiles = sorted([i for i in files if f'wrfout_{NEST}_' in i])
ncfiles = [i for i in ncfiles if '.' not in i] # remove any bad files

timecount = 0
timecountend = 0

for i in range(len(ncfiles)):
    print(ncfiles[i])
    
    #open netCDF file (ncread)
    ncfile = nc.Dataset(file_path + ncfiles[i], 'r')
    
    print(get_cartopy(wrfin=ncfile))
    
    wrf_proj = pyproj.Proj(proj='lcc', # projection type: Lambert Conformal Conic
                   lat_1=ncfile.TRUELAT1, lat_2=ncfile.TRUELAT2, # Cone intersects with the sphere
                   lat_0=ncfile.MOAD_CEN_LAT, lon_0=ncfile.STAND_LON, # Center point
                   a=6370000, b=6370000) # This is it! The Earth is a perfect sphere
    
    # Easting and Northings of the domains center point
    wgs_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    e, n = pyproj.transform(wgs_proj, wrf_proj, ncfile.CEN_LON, ncfile.CEN_LAT)
    # Grid parameters
    dx, dy = ncfile.DX, ncfile.DY
    nx, ny = ncfile.dimensions['west_east'].size, ncfile.dimensions['south_north'].size
    # Down left corner of the domain
    x0 = -(nx-1) / 2. * dx + e
    y0 = -(ny-1) / 2. * dy + n
    # 2d grid
    xx = np.arange(nx) * dx + x0
    yy = np.arange(ny) * dy + y0
    
    xx = xx / 1000
    yy = yy / 1000
    
    xx = np.round(xx).astype('int').astype('str')
    yy = np.round(yy).astype('int').astype('str')
    
    str1 = ', '
    print(str1.join(xx))
    print(str1.join(yy))
