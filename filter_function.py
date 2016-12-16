import numpy as np

def starFilter(data):
	"""
	red_clump_13_TEFF_up6011.0_lo4134.5
	"""
	return (data['TEFF'] < 6011.0) & (data['TEFF'] > 4134.5)