#!/bin/bash

#SBATCH --account=strong-kp
#SBATCH --partition=strong-kp
#SBATCH --job-name=debias_attempt
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=take1.out    
#SBATCH --error=take1.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=u1301408@utah.edu


# Similarly to the scripts from savanna, create an sbatch that loops through all of the files and pass each file to ncl scripts that actually do the work
# Create enviornment to run in

# Define locations of where to find and store data
dir_in="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/"
dir_corrected="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/corrected_dim/"
dir_out="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/debias/debias_out/"

# Set variable to perform debiasing on
# gridMET variables: tmmn, tmmx, pr, sph, srad, vas, uas (note: on 4km resolution grid)
# WRF equivalent: T2MIN, T2MAX, RAINNC+RAINC, QVAPOR (have to calculate specific humidity), SWDOWN, V, U
vname="T2MIN"
vnew="tmmn"

# Years in historical period and domain(s) of interest
# Bias removed from the dataset is based on the years in the hitorical period (1985-2014)
years=(1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014)
domains=(03)

# Loop through the datasets to correct time dimension
# first.ncl should: 
# 1. Loop through all of the six hourly data files over the historical period (1985-2014) 
# 2. Correct the time dimension in each file
# 3. Extract and rename the varible of interest -> create an if statement that calculates specific humidity from qvapor (q/q+1)
# 4. Compress the file for storage efficiency

# todo: Before running the loop for all 30 years, run it for just one file (e.g., replace the loop with for file_path in ${dir_in}/wrfout_d03_1985-01-01_00:00:00; do ...). This will verify your ncl script isn't hitting any syntax errors.

for year in "${years[@]}" ; do	# loop through year
	for domain in "${domains[@]}" ; do	# loop through domain
		files=`ls ${dir_in}/wrfout_d${domain}_${year}-*`  # pull all files

		for file in $(echo $files | tr " " "\n") ; do	# loop through all files

			#file_in=${file:75}
			#file_out=${file:75}
			file_in=${file##*/}
			file_corrected=${file##*/}_${vnew}

			#echo $dir_in
			echo $file_in
			#echo $dir_out
			#echo $file_out
		
			ncl 'dir_in="'${dir_in}'"' 'file_in="'${file_in}'"' 'dir_corrected="'${dir_corrected}'"' 'file_corrected="'${file_corrected}'.nc"' 'vname="'${vname}'"' 'vnew="'${vnew}'"' first.ncl

		done
	done
done

# Concatonate all of the corrected six hourly files together along the new time dimension (create master file)
ncrcat -O ${dir_corrected}/*${vnew}* ${dir_corrected}/${vnew}_1985-2014.nc

# Remove the old six hourly files once a master file is created
rm ${dir_corrected}/wrfout_d${domain}*${vnew}*


maybe the first ncl script is to correct the time dimension because that will be used for every variable and then a second one is created to do the rest

second ncl for conducting preprocessing
1. feed ncl all netcdf files for a given month (i.e. all january netcdf files)
2. calculate the monthly average across the historical period at each gridpoint (should produce a 3D array: time x lat x lon)
3. do the same thing for gridmet
4. interpolate gridmet to wrf grid
5. subtract gridmet from WRF (pr should be a ratio)
6. use a harmonic function to fit a smooth curve to monthly averages (there should be a unique function for each grid point)
7. from the smooth curve, derive a daily bias value for each gridpoint
8. subtract that daily value from the file produced by the first ncl script

the know how:
1. create script in ncl
2. is there a function for harmonics in ncl
3. how to build a netcdf that matches the dimensions of the first ncl file output (makes step 8 for second ncl file easy if they are the same shape)





