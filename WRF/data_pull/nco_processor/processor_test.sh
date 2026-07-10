#!/bin/bash 
#SBATCH --account=strong-kp
#SBATCH --partition=strong-kp
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=NCO_SWE
#SBATCH --output=SWE.out    
#SBATCH --error=SWE.err

# module load nco 
module load cdo

# end script if there is an error
set -e

COMBINED="SWE_combo.nc" 
FIXED="SWE_adjusted.nc"
FINAL="SWE_final.nc" 

echo "Step 1: Generating list and keeping it for inspection"
# use 'sort' to ensure the files are in chronological order for ncrcat
find /uufs/chpc.utah.edu/common/home/strong-group7/husile/gsl/wrfout_multimodel/ -name "wrfout_d03*" | sort > file_list.txt

echo "Step 2: Checking if list"
if [ ! -s file_list.txt ]; then
    echo "ERROR: No files found! Check your find path."
    exit 1
fi

echo "File list generated. First 5 files found:"
head -n 5 file_list.txt

# echo "Step 3: Concatenating files" 
# 
# cdo -f nc cat -cat file_list.txt $COMBINED
# head -n 1 file_list.txt | xargs ncrcat -O -o $COMBINED
# > $COMBINED
# cat file_list.txt | while read line; do
#     cdo -f nc cat "$line" $COMBINED
# done

echo "Step 3: Concatenating files"
# passing all file names directly to nco or cdo will cause it too crash because the list is too long
cdo -f nc cat $(cat file_list.txt) $COMBINED

# 2. Loop through the rest of the file list
#    Using a loop with a small batch size is memory efficient.
#    'sed 1d' skips the first line we just processed.
# sed '1d' file_list.txt | xargs -n 100 ncrcat -A -o $COMBINED
# echo "Step 3 complete."

echo "Step 4: Fixing time axis"
# -f nc: output format is NetCDF 
# settaxis: Set the time axis (Start date, Start time, Frequency) 
cdo -f nc settaxis,1985-01-01,00:00:00,6hours $COMBINED $FIXED

echo "Step 5: Extracting Water Year Months and save only SWE" 
# Select only months 10, 11, 12, 1, 2, 3, 4 
cdo selvar,SNOW -selmon,10,11,12,1,2,3,4 $FIXED $FINAL

# Cleanup: Remove intermediate large files to save quota 
rm $COMBINED $FIXED

echo "Done! Final file is $FINAL"
