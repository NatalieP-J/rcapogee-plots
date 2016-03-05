def starFilter(data):
    """
    Returns True where stellar properties match conditions
    
    """
    return (data['TEFF'] > 4500) & (data['TEFF'] < 4800)
