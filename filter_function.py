import numpy as np

def starFilter(data):
	"""
	red_clump_TEFF_up4900.0_lo4700.0
	"""
	return (data['TEFF'] < 4900.0) & (data['TEFF'] > 4700.0)