#!/bin/bash
#SBATCH --job-name=SWE   # Name showing up in squeue
#SBATCH --account=strong-kp    	    # Court's node (kingspeak11)
#SBATCH --partition=strong-kp
#SBATCH --nodes=1                   # Single node is best for single script multi-threading
#SBATCH --ntasks=16   
#SBATCH --mem=55G 
#SBATCH --output=SWE.out     # Standard output log file (%j dynamically inserts Job ID)
#SBATCH --error=SWE.err      # Error log file

# Exit script immediately if any command fails
set -e

# run script in selected environment
/uufs/chpc.utah.edu/common/home/strong-group7/sydney/miniforge3_envs/olympics/bin/python SWE_slice.py
