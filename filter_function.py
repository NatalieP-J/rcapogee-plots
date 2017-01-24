import numpy as np

def starFilter(data):
	"""
	red_clump_12_TEFF_up4750.0_lo4700.0
	"""
	return (data['TEFF'] < 4750.0) & (data['TEFF'] > 4700.0)