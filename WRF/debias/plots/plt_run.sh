#!/bin/bash

#SBATCH --account=strong-kp
#SBATCH --partition=strong-kp
#SBATCH --job-name=marg
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=marg.out    
#SBATCH --error=marg.err
 
# Load in software
module load miniforge3
source /uufs/chpc.utah.edu/sys/installdir/r8/miniforge3/25.11.0/etc/profile.d/conda.sh
conda activate olympics

# Point to python bin in olympics
PYTHON_BIN="/uufs/chpc.utah.edu/common/home/strong-group7/sydney/miniforge3_envs/olympics/bin/python"

# Call python file
${PYTHON_BIN} validation.py 


