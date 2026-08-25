"""
Author: Sydney Smith
Date Created: August 25, 2026
"""

import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from old.temporal_chunks import get_fpaths, interpo_MET, WRF_daily

