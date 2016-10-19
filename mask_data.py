import numpy as np
from apogee.tools import bitmask as bm
import data_access
reload(data_access)
from data_access import *
from star_sample import subStarSample

def bitsNotSet(bitmask,maskbits):
    """
    Given a bitmask, returns True where any of maskbits are set 
    and False otherwise.
    
    bitmask:   bitmask to check
    maskbits:  bits to check if set in the bitmask
    
    """
    goodLocs_bool = np.zeros(bitmask.shape).astype(bool)
    for m in maskbits:
        bitind = bm.bit_set(m,bitmask)
        goodLocs_bool[bitind] = True
    return goodLocs_bool

# Example mask filter function for APOGEE
badcombpixmask = bitmask.badpixmask()
badcombpixmask += 2**bitmask.apogee_pixmask_int("SIG_SKYLINE")
def maskFilter(sample,minstar=5):
    """
    Returns True where sample properties match conditions
    
    sample:   an object of mask class
    minstar:  minimum number of unmasked stars required at a pixel for that 
              pixel to remain unmasked

    """
    # Artificially reduce SNR by increasing uncertainty where SNR is high
    sample.spectra_errs[sample._SNR>200] = sample.spectra[sample._SNR>200]/200.
    sample._SNR = sample.spectra/sample.spectra_errs
    # Breakdown badcombpixmask (from data.py) into each individual bit flag
    maskbits = bm.bits_set(badcombpixmask)
    # Mask where SNR low or where something flagged in bitmask
    mask = (sample._SNR < 50.) | bitsNotSet(sample._bitmasks,maskbits)
    # Calculate the number of masked stars at each pixel
    flaggedstars = np.sum(mask,axis=0)
    # Flag pixels where there aren't enough stars to do the fit
    flaggedpix = flaggedstars > (sample.numberStars-minstar)
    mask.T[flaggedpix]=True
    return mask

class mask(subStarSample):
    """
    Define and apply a mask given a set of conditions in the form of the
    maskConditions function.
    
    """
    def __init__(self,dataSource,sampleType,maskFilter,ask=True):
        """
        Mask a subsample according to a maskFilter function
        
        sampleType:   designator of the sample type - must be a key in readfn 
                      and independentVariables in data.py
        maskFilter:   function that decides on elements to be masked
        ask:          if True, function asks for user input to make 
                      filter_function.py, if False, uses existing 
                      filter_function.py
        
        """
        subStarSample.__init__(self,dataSource,sampleType,ask=ask)
        self._SNR = self.spectra/self.spectra_errs
        # find indices that should be masked
        self._maskHere = maskFilter(self,minstar=5)
        self.applyMask()

    def applyMask(self):
        """
        Mask all arrays according to maskConditions

        """
        # create default mask arrays (mask nothing)
        self.masked = np.zeros(self.spectra.shape).astype(bool)
        self.unmasked = np.ones(self.spectra.shape).astype(bool)
        # update mask arrays
        self.masked[self._maskHere]= True
        self.unmasked[self._maskHere] = False
        # apply mask arrays to data
        # spectral information
        self.spectra.mask = self.masked
        self.spectra_errs.mask = self.masked

        # create dictionary tracing the independent variables to keywords
        self.keywordMap = {'TEFF':self.teff,
                           'LOGG':self.logg,
                           'FE_H':self.fe_h
                       }

