#!/bin/bash

#SBATCH --cluster=notchpeak
#SBATCH --account=uspcasw-np
#SBATCH --partition=uspcasw-np
#SBATCH --job-name=pr
#SBATCH --time=3-00:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=pr.out    
#SBATCH --error=pr.err
 
# Load in software
module load miniforge3
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate olympics

# Define variables
export variable="pr"
export domain="03"
export WRF_in="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/"
export MET_in="/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/"
export save=True
export bias_type="multiplicative" # Options are "multiplicative" and "additive" but only pr and sph are "multipicative"
export running_window=31
export dry_day_threshold=0.01 # TODO: check units for precipitation and log sph ddt somewhere

# Point to python bin in olympics
PYTHON_BIN="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/miniforge3_envs/olympics/bin/python"

# Call python file
${PYTHON_BIN} preprocessing.py 

module load cdo

echo "Starting concatenation for ${var}..."
cdo cat daily/${var}/*.nc wrfout/wrf_raw_GSLBIP_multimodel_ssp245_${var}_cdo.nc
echo "Concatenation complete!"

# Call python file
# ${PYTHON_BIN} ecdfm_man.py 



