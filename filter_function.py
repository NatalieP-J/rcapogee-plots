import numpy as np

def starFilter(data):
	"""
	red_clump_FE_H_up-0.3_lo-0.4
	"""
	return (data['FE_H'] < -0.3) & (data['FE_H'] > -0.4)