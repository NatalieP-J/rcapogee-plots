import numpy as np

def starFilter(data):
	"""
	red_giant_12_LOGG_up4.0_lo3.0
	"""
	return (data['LOGG'] < 4.0) & (data['LOGG'] > 3.0)