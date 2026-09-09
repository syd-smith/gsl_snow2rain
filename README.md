# Climate Analysis of Utah as a Winter Olympic and Paralympic Host

## Project Description
With the International Olympic Committee considering Salt Lake City, Utah to be in permanent rotation as a Winter Olympic host, climate change poses a critical challenge to long term feasibility of the games. Unseasonably warm temperatures during the winter of 2025/26 emphasized this vulnerability. Standard global climate models (GCMs) used for climate analysis feature grids with such coarse resolution that elevation differences between the Salt Lake Valley floor and the peaks of the Wasatch Front are hardly evident. Work from Professor Strong’s research group for the Great Salt Lake Basin Integrated Plan (GSLBIP) has developed region specific datasets downscaled to 4km resolution. These data – paid for by Utah taxpayers – will be analyzed to provide direction on safely hosting the Winter Olympics while enhancing return on the state’s previous investment. Specifically, this study evaluates reliability of snow conditions from decade to decade under multiple greenhouse gas emission scenarios to determine a statistical probability of event locations passing the International Olympic Committee’s hosting requirements. Research also assesses the viability of climate adaptation strategies such as snow storage from year to year. This project leverages local expertise to provide state leadership with a rigorous, data driven framework to guide future Olympic planning.

## Folder Structure
```
.
├──LAKE_NO_LAKE                    # WRF simulations over North Utah with and without the Great Salt Lake
├──MACA                            # Analysis of statistically downscaled data (over 100 emission scenarios)
│   └── first_look                 # Initial analysis of MACA data using pressure interpolation
├──WRF                             # Analysis of dynamically downscaled WRF data using debiased GCM data for boundary conditions
|   ├── data_processing_attempts                 
|   ├── debias                     
|   |   ├── debias.sh              # Bash script to debias WRF outputs (controls spatial_chunk.py where the debiasing code is housed)
│   |   └── plots
│   |       └── validation.py      # Plotting code to validate debiasing process was performed correctly     
│   └── proposal                   # Initial look at WRF data
└──from_savanna                    # Scripts from Savanna
README.md
```

## Datasets
#### Statistically Downscaled Data
Multivariate Adaptive Constructed Analogs version 2 [(MACAv2)](https://www.climatologylab.org/maca.html) is a statistical method for downscaling Global Climate Models (GCMs) from their native coarse resolution to a higher spatial resolution that captures reflects observed patterns of daily near-surface meteorology and simulated changes in GCMs experiments (Abatzoglou et al., 2012). This method has been shown to be slightly preferable to direct daily interpolated bias correction in regions of complex terrain due to its use of a historical library of observations and multivariate approach. 

The Coupled Model Intercomparison Project Phase 6 (CMIP6) multi-model ensemble (Eyring et al., 2016) dataset was chosen for downscaling by the Utah Department of Natural Resources as a part of the GSLBIP. The MACAv2 method was used on CMIP6 data to increase its spatial resolution to a fine grid using observational data from [gridMET](https://www.climatologylab.org/gridmet.html). Please visit the [MACAv2 repository](https://github.com/s-wolvin/MACAv2-METDATA_CMIP6) created by Savanna Wolvin for this project for more information.

#### Dynamically Downscaled Data

## References
Abatzoglou J.T. and Brown T.J. A comparison of statistical downscaling methods suited for wildfire applications, International Journal of Climatology (2012), 32, 772-780
