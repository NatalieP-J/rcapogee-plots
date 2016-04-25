import numpy as np

def starFilter(data):
	"""
	red_clump_TEFF_up6268.69189453_lo5000.0
	"""
	return (data['TEFF'] < 6268.69189453) & (data['TEFF'] > 5000.0)