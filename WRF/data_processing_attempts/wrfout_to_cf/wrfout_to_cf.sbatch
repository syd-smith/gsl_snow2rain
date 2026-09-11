#!/bin/bash

#SBATCH --account=strong
#SBATCH --partition=kingspeak
#SBATCH --job-name=ncl_pull
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=ncl_stable.out    
#SBATCH --error=ncl_stable.err
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=u1301408@utah.edu


# Preset Variables
dir_in="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/wrfout_multimodel_hist_1984-2014/"
dir_out="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/olympics/WRF/wrfout/"
domains=(03)
years=(1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014)


# Load in software
module load miniforge3
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate ncl_stable

# Force Linux to look in conda environment's lib/ folder first before default CHPC library versions
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


# Loop through the datasets
for year in "${years[@]}" ; do	# loop through year
	for domain in "${domains[@]}" ; do	# loop through domain
		files=`ls ${dir_in}/wrfout_d${domain}_${year}-*`  # pull all files

		for file in $(echo $files | tr " " "\n") ; do	# loop through all files

			#file_in=${file:75}
			#file_out=${file:75}
			file_in=${file##*/}
			file_out=${file##*/}

			#echo $dir_in
			echo $file_in
			#echo $dir_out
			#echo $file_out
		
			ncl 'dir_in="'${dir_in}'"' 'file_in="'${file_in}'"' 'dir_out="'${dir_out}'"' 'file_out="'${file_out}'.nc"' wrfout_to_cf.ncl

		done
	done
done








