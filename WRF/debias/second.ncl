;1. feed ncl all netcdf files for a given month (i.e. all january netcdf files)
;2. calculate the monthly average across the historical period at each gridpoint (should produce a 3D array: time x lat x lon)
;3. do the same thing for gridmet
;4. interpolate gridmet to wrf grid
;5. subtract gridmet from WRF (pr should be a ratio)
;6. use a harmonic function to fit a smooth curve to monthly averages (there should be a unique function for each grid point)
;7. from the smooth curve, derive a daily bias value for each gridpoint
;8. subtract that daily value from the file produced by the first ncl script