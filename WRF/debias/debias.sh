#!/bin/bash

#SBATCH --cluster=notchpeak
#SBATCH --account=uspcasw-np
#SBATCH --partition=uspcasw-np
#SBATCH --job-name=sph
#SBATCH --time=3-00:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=sph.out    
#SBATCH --error=sph.err
 
# Load in software -> might need to change depending on cluster
module load miniforge3 # Uses conda
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate debias_env

# Set variable based on gridMET variable save names
# tmmn, tmmx, pr, sph, srad, vas, uas

# Define variables
export variable="sph"
export domain="03"
export WRF_in="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/"
export MET_in="/uufs/chpc.utah.edu/common/home/strong-group7/savanna/maca/gridmet/"
export save=True
export bias_type="multiplicative" # Options are "multiplicative" and "additive" but only pr and sph are "multipicative"
export day_running_window=31 # 15 days on either side of the date in question (matches MACA's 15 day semi window)
export dry_day_threshold=6.6e-5 # PR:0.254 mm (0.01 in) & SPH:6.6e-5 kg kg-1

# Point to python bin in olympics
PYTHON_BIN="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/miniforge3_envs/debias_env/bin/python"

# Call python file
${PYTHON_BIN} preprocessing.py 

# Call python file
${PYTHON_BIN} qdm_man.py 



