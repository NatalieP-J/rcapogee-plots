import numpy as np

def starFilter(data):
	"""
	red_clump_TEFF_up6268.69189453_lo4130.89257812
	"""
	return (data['TEFF'] < 6268.69189453) & (data['TEFF'] > 4130.89257812)
