#!/bin/bash 
#SBATCH --account=strong-kp
#SBATCH --partition=strong-kp
#SBATCH --nodes=1 
#SBATCH --ntasks=16
#SBATCH --mem=50G 
#SBATCH --job-name=NCO_SWE
#SBATCH --output=SWE.out    
#SBATCH --error=SWE.err

module load nco 
module load cdo

INPUT_PATH="/uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/*/wrfout_d03*" 
COMBINED="SWE_combined.nc" 
FIXED="SWE_time_fixed.nc"
FINAL="SWE_slice_wateryear.nc" 

echo "Step 1: Concatenating..." 
ncrcat $INPUT_PATH $COMBINED 

echo "Step 2: Fixing time axis…” 
# -f nc: output format is NetCDF 
# settaxis: Set the time axis (Start date, Start time, Frequency) 
cdo -f nc settaxis,1985-01-01,00:00:00,6hours $COMBINED $FIXED

echo "Step 3: Extracting Water Year Months (Oct 1 - April 1)..." 
# Select only months 10, 11, 12, 1, 2, 3, 4 
cdo selmon,10,11,12,1,2,3,4 $FIXED $FINAL

# Cleanup: Remove intermediate large files to save quota 
rm $COMBINED $FIXED

echo "Done! Final file is $FINAL"
