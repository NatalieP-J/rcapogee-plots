def starFilter(data):
	"""
	red_clump_TEFF_up4850.0_lo4830.0
	"""
	return (data['TEFF'] < 4850.0) & (data['TEFF'] > 4830.0)