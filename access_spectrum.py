import apogee.tools.read as apread
import numpy as np
import pickle

def get_spectra_asp(data,ext = 1):
    """
    Returns aspcapStar spectra and header information for each object specified in data 
    
    data:    labels for a subset of the APOGEE survey
    """
    specs = np.zeros((len(data),7214),dtype = np.float32)
    hdrs = {}
    goodind = []
    badind = []
    for i in range(len(data)):
        try:
            specs[i] = apread.aspcapStar(data['LOCATION_ID'][i],data['APOGEE_ID'][i],ext = ext, header = False, aspcapWavegrid=True)
            goodind.append(i)
        except IOError as e:
            badind.append(i)
            print i,e
            continue
    if badind == []:
        return specs
    if badind != []:
        return (specs,(np.array(goodind),))


def get_spectra_ap(data,ext = 1,indx = None):
    """
    Returns apStar spectra and header information for each object specified in data 
    
    data:    labels for a subset of the APOGEE survey
    """
    specs = np.zeros((len(data),7214),dtype=np.int16)
    hdrs = {}
    goodind = []
    badind = []
    for i in range(len(data)):
        try:
            specs[i] = apread.apStar(data['LOCATION_ID'][i],data['APOGEE_ID'][i],ext = ext, header = False, aspcapWavegrid=True)[indx]
            goodind.append(i)
        except IOError as e:
            badind.append(i)
            print i,e
            continue
    if badind == []:
        return specs
    if badind != []:
        return (specs,(np.array(goodind),))

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
