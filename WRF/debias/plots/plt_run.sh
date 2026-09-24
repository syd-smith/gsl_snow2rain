#!/bin/bash

#SBATCH --account=uspcasw-np
#SBATCH --partition=uspcasw-np
#SBATCH --job-name=cdf
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=cdf.out    
#SBATCH --error=cdf.err
 
# Load in software
module load miniforge3
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate olympics

# Point to python bin in olympics
PYTHON_BIN="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/miniforge3_envs/olympics/bin/python"

# Call python file
${PYTHON_BIN} validation.py 


