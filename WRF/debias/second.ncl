;1. call master netcdf file
;2. calculate the monthly average across the historical period at each gridpoint (should produce a 3D array: time x lat x lon)
;3. do the same thing for gridmet
;4. interpolate gridmet to wrf grid
;5. subtract gridmet from WRF (pr should be a ratio)
;6. use a harmonic function to fit a smooth curve to monthly averages (there should be a unique function for each grid point)
;7. from the smooth curve, derive a daily bias value for each gridpoint
;8. subtract that daily value from the file produced by the first ncl script

; Ensure the passed-in string is treated as string
vnew  := tostring(vnew)

; 1. Open input files (WRF and gridMET) as read only to not overwrite any original data
f_in = addfile(dir_corrected + file_corrected, "r")
wrf_var = f_in->vnew
gMET_in = addfile(gMET_in + gMET_file, "r")
gMET_var = gMET_in->vnew

; 2. Create output files to save new data in
f_out = addfile(dir_out + file_out, "c")

collapse to daily not monthly
; 3. Calculate the monthly average for both datasets
wrf_monthly = calculate_monthly_values(wrf_var, "avg", 0, False)
gMET_monthly = calculate_monthly_values(gMET_var, "avg", 0, False)

; 4. Calculate the long-term monthly climatology across the historical period
wrf_clm = clmMonTLL(wrf_monthly)
gMET_clm = clmMonTLL(gMET_monthly)
; This show create a 3D array with the dimensions (time, lat, lon) where time has 12 values (one for each month of the year)

; 5. Interpolate gridMET to the same grid as WRF








; https://www.ncl.ucar.edu/Document/Functions/Built-in/ezfftf.shtml - returns fourier coefficents for a harmonic function
; If npts is the size of the dimension to be transformed, and N represents all other variable dimensions, then the fourier coefficients are returned as an variable dimensioned 2 x N x (npts/2). The leftmost dimension of size 2 contains the real and imaginary coefficients
; The function ezfftf (Easy Fast Fourier Transform) is the most common tool in NCL for this. It decomposes your 12-month series into a set of coefficients.
; You can then keep only the first few harmonics (e.g., the first 2 or 3) to represent the "smooth" seasonal cycle, discarding the higher-frequency noise.


; https://www.ncl.ucar.edu/Document/Functions/Built-in/ezfftb.shtml - returns harmonic curve if passed first harmonic coefficients
; Use ezfftb (Easy Fast Fourier Transform Backward) to reconstruct the smooth curve from those selected coefficients.
; all other coeeficients beside those for the first harmoic should be set to zero


