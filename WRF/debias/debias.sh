#!/bin/bash

#SBATCH --account=strong
#SBATCH --partition=kingspeak
#SBATCH --job-name=uas
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=uas.out    
#SBATCH --error=uas.err
 
# Load in software
module load miniforge3
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate olympics

# Point to python bin in olympics
PYTHON_BIN="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/miniforge3_envs/olympics/bin/python"

# Define locations of where to find and store WRF data
dir_in="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/"
dir_corrected="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/corrected_dim/"
dir_out="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/debias_out/"

# Define location of gridMET data
gMET_in="/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/gsl_region_sph_1979-2014.nc"
gMET_out="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/gMET_concat/"

# Set variable to perform debiasing on
# gridMET variables: tmmn, tmmx, pr, sph, srad, vas, uas (note: on 4km resolution grid)
# WRF equivalent: T2MIN, T2MAX, RAINNC, Q2, SWDOWN, V10, U10
vname="U10" # Should be the variable name that WRF recognizes
vnew="uas" # Should be the variable name that gridMET recognizes

# Loop through the datasets to correct time dimension
# first.ncl should: 
# 1. Loop through all of the six hourly data files over the historical period (1985-2014) 
# 2. Correct the time dimension in each file
# 3. Extract and rename the varible of interest -> create an if statement that calculates specific humidity from qvapor (q/q+1)
# 4. Compress the file for storage efficiency

# Domain(s) of interest
domain="03"

# Check if file_list.txt already exists
if [ -f "file_list.txt" ]; then
    echo "Found existing file_list.txt"
else
    echo "Generating new file list"
    find "${dir_in}" -maxdepth 1 -name "wrfout_d${domain}_*" | sort > file_list.txt
fi

# Call python file to make corrected files
${PYTHON_BIN} first.py \
    --dir_corrected "${dir_corrected}" \
    --vname "${vname}" \
    --vnew "${vnew}"

# Concatonate all of the corrected six hourly files together along the new time dimension (create master file)
ncrcat -O ${dir_corrected}${vnew}/clean* ${dir_corrected}${vnew}_1985-2014.nc
echo "Files concatonated"

# Remove the old six hourly files once a master file is created
# rm -r ${dir_corrected}${vnew}/
# echo "Six hourly files removed"


# todo: create loop for second.ncl to call files (is a loop even needed?)

# second ncl for conducting preprocessing
# 1. call master netcdf file
# 2. calculate the monthly average across the historical period at each gridpoint for each month (should produce a 3D array: time (12) x lat x lon) clmMonTLL
# 3. do the same thing for gridmet 
# 4. interpolate gridmet to wrf grid (area_conserve_remap_Wrap)
# 5. subtract gridmet from WRF (pr should be a ratio)
# 6. use a harmonic function to fit a smooth curve to monthly averages (there should be a unique function for each grid point)
# 7. from the smooth curve, derive a daily bias value for each gridpoint (clmMon2clmDay)
# 8. subtract that daily value from the file produced by the first ncl script (array level subtraction)

# the know how:
# 1. create script in ncl
# 2. is there a function for harmonics in ncl
# 3. how to build a netcdf that matches the dimensions of first.ncl file output (makes step 8 for second ncl file easy if they are the same shape)






