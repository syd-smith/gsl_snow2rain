#!/bin/bash

#SBATCH --cluster=notchpeak
#SBATCH --account=uspcasw-np
#SBATCH --partition=uspcasw-np
#SBATCH --job-name=cdf
#SBATCH --time=1-00:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=cdf.out    
#SBATCH --error=cdf.err

# Define variables
export variable="pr"
export domain="03"
export WRF_in="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/"
export MET_in="/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/"
export save=True
export bias_type="multiplicative" # Options are "multiplicative" and "additive" but only pr and sph are "multipicative"
export running_window=31 # 15 days on either side of the date in question (matches MACA's 15 day semi window)
export dry_day_threshold=0.254 # PR: 0.254 mm & SPH:6.6e-5 kg kg-1

# Load in software
module load miniforge3
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate olympics

# Point to python bin in olympics
PYTHON_BIN="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/miniforge3_envs/olympics/bin/python"

# Call python file
${PYTHON_BIN} validation.py 


