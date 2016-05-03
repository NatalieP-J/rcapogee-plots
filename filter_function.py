import numpy as np

def starFilter(data):
	"""
	red_clump_FE_H_up-0.1_lo-0.105
	"""
	return (data['FE_H'] < -0.1) & (data['FE_H'] > -0.105)