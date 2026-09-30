"""
Author: Sydney Smith
Date Created: September 30, 2026
"""

from datetime import datetime
from loguru import logger
import os
import sys
import xarray as xr
from xsdba import Grouper
from xsdbs.adjustment import EmpiricalQuantileMapping as EQM

# ==================================
# - Establish Relative File Path - 
# ==================================

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from plots.validation import running_window_slice, loc_sel

# ===================
# - Set Up Logger - 
# ==================

# Create directory for log files if it doesn't already exist
log_path = str(current_dir / 'log')
os.makedirs(log_path, exist_ok = True)

# String filename
log_filename = datetime.now().strftime('log/%Y-%m-%d_%H-%M-%S.log')

# Custom format for log prints
log_format = log_format = '<cyan>{time:HH:mm:ss}</cyan> | <level>{level:>8}</level> | <yellow>{name}</yellow>:<cyan>{function}</cyan>:line <magenta>{line}</magenta> - <level>{message}</level>'

# Removes the default stderr sink
logger.remove()  
# Anything above info to console
logger.add(sys.stderr, colorize = True, format = log_format, level = 'INFO') 
# Anything above debug to log file
logger.add(log_filename, colorize = False, format = log_format, level = 'DEBUG')
logger.debug(f'Log files saved to {log_path}')

# =====================
# ---- Functions ----
# =====================

def debiaser(obs, hist, raw):

    # Create running windows for distribution grouping
    custom_grouper = Grouper(group = 'time.dayofyear', window = 31)

    # Contruct debiaser object
    eqm = EQM(
        nquantiles = 50, # bins distribution to have a certain number of quantiles
        kind = '+', # Additive vs. multplicative
        group = custom_grouper
        )

    # Train debiaser using reference and historical data
    eqm.train(ref = obs, hist = hist)

    # Bias correct the raw model data
    debiased_data = eqm.adjust(raw)