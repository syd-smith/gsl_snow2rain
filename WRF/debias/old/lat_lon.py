# %%
"""
Author: Sydney Smith
Date Created: July 30, 2026
"""

import geopandas as gpd

# Load in the shape file that contains the new boundaries for the GSLB
shp  = gpd.read_file('/uufs/chpc.utah.edu/common/home/strong-group7/sydney/GSLBIP/from_savanna/WBD_16_HU2_Shape/Shape/WBDHU4.shp')
gsl  = shp[shp["huc4"] == "1602"]
br   = shp[shp["huc4"] == "1601"]
gslb = gpd.GeoDataFrame(geometry=[gsl.geometry.unary_union.union(br.geometry.unary_union)], crs=shp.crs)

print(gslb.total_bounds)
# array([-115.00992021,   37.94716302, -110.59847552,   42.85578334])
# %%
