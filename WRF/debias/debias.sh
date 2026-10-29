#!/bin/bash

#SBATCH --cluster=notchpeak
#SBATCH --account=uspcasw-np
#SBATCH --partition=uspcasw-np
#SBATCH --job-name=tmmx
#SBATCH --time=3-00:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=tmmx.out    
#SBATCH --error=tmmx.err
 
# Load in software
module load miniforge3
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate olympics

# Set variable based on gridMET variable save names
# tmmn, tmmx, pr, sph, srad, wind (vas & uas)

# Define variables
export variable="tmmx"
export domain="03"
export WRF_in="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/"
export MET_in="/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/"
export save=True
export bias_type="additive" # Options are "multiplicative" and "additive" but only pr and sph are "multipicative"
export day_running_window=31 # 15 days on either side of the date in question (matches MACA's 15 day semi window)
export year_running_window=31
export dry_day_threshold=0.254 # PR: 0.254 mm & SPH:6.6e-5 kg kg-1

# Point to python bin in olympics
PYTHON_BIN="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/miniforge3_envs/olympics/bin/python"

# Call python file
# ${PYTHON_BIN} preprocessing.py 

# module load cdo

# echo "Starting concatenation for ${var}..."
# cdo cat daily/${var}/*.nc wrfout/wrf_raw_GSLBIP_multimodel_ssp245_${var}_cdo.nc
# echo "Concatenation complete!"

# Call python file
${PYTHON_BIN} ecdfm_man.py 



