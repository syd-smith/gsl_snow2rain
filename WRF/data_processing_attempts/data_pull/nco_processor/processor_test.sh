#!/bin/bash 
#SBATCH --account=strong-kp
#SBATCH --partition=strong-kp
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=NCO_SWE
#SBATCH --output=SWE.out    
#SBATCH --error=SWE.err

module load miniforge3
# environment has cdo and nco installed
conda activate nco_env

# end script if there is an error
set -e

FINAL="SWE_final.nc" 

echo "Step 1: Generating list and keeping it for inspection"
# use 'sort' to ensure the files are in chronological order for ncrcat
find /uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/ -name "wrfout_d03*" | sort > file_list.txt

# add files with proper time dimensions to a new list and print names of files that have a time dimension of 0
for line in $(cat file_list.txt); do
    # Check if the time dimension is NOT 0
    if ncdump -h "$line" | grep -q "time = 0"; then
        echo "$line" >> clean_file_list.txt
    else
        echo "Filtering out bad file: $line"
    fi
done

echo "Step 2: Chunk file list into smaller pieces"
split -l 500 clean_file_list.txt chunk_

# Loop through the chunks and merge each into an intermediate file
for chunk in chunk_*; do
    # Read the filenames in the chunk and pass them to ncrcat
    ncrcat -v SNOW $(cat $chunk) "temp_${chunk}.nc"
done

# Merge all the intermediate files into your final output
ncrcat temp_chunk_*.nc $FINAL

# Clean up the mess
rm temp_chunk_*.nc chunk_*

echo "Step 3: Fixing time axis"
# Note: You can apply this to the final file if needed
cdo -f nc settaxis,1985-01-01,00:00:00,6hours $FINAL temp_fixed.nc
mv temp_fixed.nc $FINAL

echo "Done! Final file is $FINAL"
