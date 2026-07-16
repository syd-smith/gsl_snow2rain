/;
Author: Sydney Smith
Date: July 16, 2026
;/

; This script should: 
; 1. Loop through all of the six hourly data files over the historical period (1985-2014) 
; 2. Correct the time dimension in each file
; 3. Extract and rename the varible of interest -> create an if statement that calculates specific humidity from qvapor (q/q+1)
; 4. Compress the file for storage efficiency


; --- Compression Settings ---
; Must be set before opening the output file
setfileoption("nc", "Format", "NetCDF4")
setfileoption("nc", "CompressionLevel", 5) 
setfileoption("nc", "Shuffle", 1)

; Ensure the passed-in strings are treated as strings
vname := tostring(vname)
vnew  := tostring(vnew)

; 1. Open input files as read only to not overwrite any original data
f_in = addfile(dir_in + file_in, "r")

; 2. Create output files to save new data in
f_out = addfile(dir_corrected + file_corrected, "c")

; 3. Setup the UNLIMITED dimension in the NEW file to be able to concatonate along time dimension
filedimdef(f_out, "time", -1, True)

; 4. Convert Times string to numeric time coordinate
Times = f_in->Times
time_numeric = wrf_times_c(Times, 0)
time_numeric@units = "hours since 1984-01-01 00:00:00"
time_numeric@calendar = "gregorian"
time_numeric!0 = "time"
time_numeric&time = time_numeric

; 5. Write corrected time dimension to f_out
; -> used to import or export variables via addfiles function
f_out->time = time_numeric

if (vname .eq. "QVAPOR") then

    ; 6. Pull out the variable of interest
    qv = f_in->QVAPOR

    ; 7. Calculate specific humidity: q = qv / (1 + qv)
    ; QVAPOR units are kg kg-1
    new_var = qv / (1.0 + qv) 
    copy_VarMeta(qv, new_var)
    new_var@long_name = "Specific Humidity"

    ; 8. Adjust dimensions for the variable
    new_var!0 = "time"
    new_var&time = time_numeric

    ; 8. Write to f_out using a dynamic string pointer to new variable name from sbatch script
    f_out->$vnew$ = new_var

else

    ; 6. Pull out the variable of interest
    ; Read the variable from the input file using the dynamic string from sbatch script
    temp_var = f_in->$vname$

    ; 7. Adjust dimensions for the variable
    temp_var!0 = "time"
    temp_var&time = time_numeric

    ; 8. Write to f_out using a dynamic string pointer to new variable name from sbatch script
    ; f_out->$vnew$ tells NCL to name the variable the new name in the file
    f_out->$vnew$ = temp_var

end if





