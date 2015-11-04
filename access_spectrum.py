import apogee.tools.read as apread
import numpy as np
import pickle

def get_spectra_asp(data,ext = 1,header=False):
    """
    Returns aspcapStar spectra and header information for each object specified in data 
    
    data:    labels for a subset of the APOGEE survey
    """
    specs = []
    hdrs = {}
    goodind = []
    badind = []
    for i in range(len(data)):
        try:
            spec = apread.aspcapStar(data['LOCATION_ID'][i],data['APOGEE_ID'][i],ext = ext, header = header, aspcapWavegrid=True)
            goodind.append(i)
            if header:
                spec,hdr = spec
                hdrs[i] = hdr
            specs.append(spec)
        except IOError as e:
            badind.append(i)
            print e
            continue
    if badind == []:
        if header:
            return np.array(specs),hdrs
        elif not header:
            return np.array(specs)
    elif badind != []:
        if header:
            return np.array(specs),hdrs,goodind
        elif not header:
            return np.array(specs),goodind

def get_spectra_ap(data,ext = 1, header = False,indx = None):
    """
    Returns apStar spectra and header information for each object specified in data 
    
    data:    labels for a subset of the APOGEE survey
    """
    specs = []
    hdrs = {}
    for i in range(len(data)):
    	if indx != None and header == False:
    		spec = apread.apStar(data['LOCATION_ID'][i],data['APOGEE_ID'][i],ext = ext, header = header, aspcapWavegrid=True)[indx]
    	elif indx == None or header != False:
        	spec = apread.apStar(data['LOCATION_ID'][i],data['APOGEE_ID'][i],ext = ext, header = header, aspcapWavegrid=True)
	        if header:
	            spec,hdr = spec
	            hdrs[i] = hdr
        specs.append(spec)
    if header and indx == None:
        return np.array(specs),hdrs
    elif not header:
        return np.array(specs)

def pklread(fname):
    """
    Opens a pickled file with name fname.
    
    Returns data array stored in pickled file.
    """
    pklffile = open(fname,"rb")
    dat = pickle.load(pklffile)
    pklffile.close()
    return dat

def pklwrite(fname,dat):
    """
    Pickles dat and writes it to file with name fname.
    """
    pklffile = open(fname,"wb")
    pickle.dump(dat,pklffile)
    pklffile.close()
