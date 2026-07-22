#!/bin/bash

#SBATCH --account=strong
#SBATCH --partition=kingspeak
#SBATCH --job-name=debias_attempt
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=test1.out    
#SBATCH --error=test1.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=u1301408@utah.edu

# Load in software
module load miniforge3
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate ncl_stable

# Define locations of where to find and store WRF data
dir_in="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/"
dir_corrected="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/corrected_dim/"
dir_out="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/debias_out/"

# Define location of gridMET data
gMET_in="/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/"
gMET_out="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/gMET_concat/"

# Set variable to perform debiasing on
# gridMET variables: tmmn, tmmx, pr, sph, srad, vas, uas (note: on 4km resolution grid)
# WRF equivalent: T2MIN, T2MAX, RAINNC+RAINC, Q2 (have to calculate specific humidity), SWDOWN, V10, U10
vname="Q2"
vnew="sph"
# T2: (Time, south_north, west_east) -> what about min and max
# Q2: (Time, south_north, west_east)
# RAINNC: (Time, south_north, west_east)
# SWDOWN: (Time, south_north, west_east) -> DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE
# V10: (Time, south_north, west_east)
# U10: (Time, south_north, west_east)


# Domain(s) of interest
domain="03"

# Loop through the datasets to correct time dimension
# first.ncl should: 
# 1. Loop through all of the six hourly data files over the historical period (1985-2014) 
# 2. Correct the time dimension in each file
# 3. Extract and rename the varible of interest -> create an if statement that calculates specific humidity from qvapor (q/q+1)
# 4. Compress the file for storage efficiency

# Define prev_file for first file (there is none)
prev_file=""

while IFS= read -r file; do

    # Define the filename
    filename=$(basename "$file")
    out_filename="corr_${filename}"

    # Bias removed from the dataset is based on the years in the hitorical period (1985-2014)
    # Call NCL once per file passing both current and previous filenames
    ncl "f_path_in=\"${file}\"" \
        "prev_file_in=\"${prev_file}\"" \
        "dir_corrected=\"${dir_corrected}\"" \
        "file_corrected=\"${out_filename}\"" \
        "vname=\"${vname}\"" \
        "vnew=\"${vnew}\"" \
        first.ncl
        
	# Store current filename as previous for the next iteration
    prev_file="${file}"
    
done < file_list.txt

# Concatonate all of the corrected six hourly files together along the new time dimension (create master file)
ncrcat -O ${dir_corrected}${vnew}/corr_wrfout* ${dir_corrected}${vnew}_1985-2014.nc
echo "Files concatonated"

# Remove the old six hourly files once a master file is created
rm -r ${dir_corrected}${vnew}/
echo "Six hourly files removed"

# Concatonate gridMET files together along time dimension
ncrcat -O ${gMET_in}*${vnew}* ${gMET_out}${vnew}_1985-2014.nc
echo "gridMET files successfully concatonated!"


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






