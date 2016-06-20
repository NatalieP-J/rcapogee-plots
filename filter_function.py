import numpy as np

def starFilter(data):
	"""
	red_clump_TEFF_up4800.0_lo4700.0
	"""
	return (data['TEFF'] < 4800.0) & (data['TEFF'] > 4700.0)