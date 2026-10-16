#!/bin/bash

#SBATCH --cluster=notchpeak
#SBATCH --account=uspcasw-np
#SBATCH --partition=uspcasw-np
#SBATCH --job-name=tmmn
#SBATCH --time=3-00:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=tmmn.out    
#SBATCH --error=tmmn.err
 
# variable="tmmn"
# domain="03"
# WRF_in="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/"
# MET_in="/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/"
# save=True


# Load in software
module load miniforge3
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate olympics

# Point to python bin in olympics
PYTHON_BIN="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/miniforge3_envs/olympics/bin/python"

# Call python file
# ${PYTHON_BIN} spatial_chunks.py 

module load cdo

echo "Starting concatenation for ${var}..."
cdo cat daily/${var}/*.nc wrfout/wrf_raw_GSLBIP_multimodel_ssp245_${var}_cdo.nc
echo "Concatenation complete!"

# Call python file
${PYTHON_BIN} ecdfm_man.py 



