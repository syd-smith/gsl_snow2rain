"""
Author: Sydney Smith
Date Created: August 25, 2026
"""

import matplotlib.pyplot as plt
from pathlib import Path
import sys
import xarray as xr

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

def trend_plt():

def cdf_plt():
    obs, raw hist, raw fut, debiased hist and fut (5 cdfs total)

def annual_scatter():
    

def main():
    # Open debiased data
    debiased = xr.open_dataset

# ======================
# ---- Entry Point ----
# ======================

if __name__ == '__main__':
    main(
        var = 'tmmn'
    )

