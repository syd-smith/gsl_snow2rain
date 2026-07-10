#!/bin/bash
#SBATCH --account=strong-kp
#SBATCH --partition=strong-kp
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=meta_check
#SBATCH --output=meta_check.out    
#SBATCH --error=meta_check.err
# Check for metadata consistency against the first file in the list

FIRST_FILE=$(head -n 1 file_list.txt)
echo "Comparing all files to: $FIRST_FILE"

# Extract variable names from the first file as a reference
ncdump -h "$FIRST_FILE" | grep -v 'dimensions' | grep 'variable' | sort > ref_vars.txt

while read line; do
    # Extract variable names from the current file
    ncdump -h "$line" | grep -v 'dimensions' | grep 'variable' | sort > current_vars.txt
    
    # Compare
    if ! cmp -s ref_vars.txt current_vars.txt; then
        echo "MISMATCH FOUND: $line"
        diff ref_vars.txt current_vars.txt
    fi
done < file_list.txt

rm ref_vars.txt current_vars.txt