# rcapogee-plots

Home to code for spectra analysis of red clump stars from the APOGEE survey.

# How to Use

Examples of how to use this code are shown in the two ipython notebooks 
(cluster_EMPCA.ipynb and red_clump_EMPCA.iypnb). Basic modifications can be easily implemented by changing data.py, which all stores a few useful functions.

# Main Functions

Main functions reside in star_sample.py, mask_data.py and empca_residuals.py.
These files contain classes to read in a sample of stellar spectra, mask it 
based on a function and fit in independent variables based on outline in 
data.py
 