import numpy as np
from apogee.tools import bitmask as bm
from spectralspace.sample.star_sample import subStarSample

# For reference, the APOGEE_PIXMASK

APOGEE_PIXMASK={0:"BADPIX", # Pixel marked as BAD in bad pixel mask            
                1:"CRPIX", # Pixel marked as cosmic ray in ap3d                
                2:"SATPIX", # Pixel marked as saturated in ap3d                
                3:"UNFIXABLE", # Pixel marked as unfixable in ap3d             
                4:"BADDARK", # Pixel marked as bad as determined from dark frame
                5:"BADFLAT", # Pixel marked as bad as determined from flat frame
                6:"BADERR", # Pixel set to have very high error (not used)     
                7:"NOSKY", # No sky available for this pixel from sky fibers   
                8:"LITTROW_GHOST", # Pixel falls in Littrow ghost              
                9:"PERSIST_HIGH", # Pixel falls in high persistence region     
                10:"PERSIST_MED", # Pixel falls in medium persistence region   
                11:"PERSIST_LOW", # Pixel falls in low persistence region      
                12:"SIG_SKYLINE", # Pixel falls near sky line                  
                13:"SIG_TELLURIC", # Pixel falls near telluric line           
                14:"NOT_ENOUGH_PSF", # Less than 50 percent PSF in good pixels
                15:"POORSNR", # Signal to noise below limit                    
                16:"FAILFIT" # Fitting for stellar parameters failed on pixel  
                }

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
badcombpixmask = bm.badpixmask()
badcombpixmask += 2**bm.apogee_pixmask_int("SIG_SKYLINE")

def maskFilter(sample,minstar=5,badcombpixmask=4351,minSNR=50.):
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
    mask = (sample._SNR < minSNR) | bitsNotSet(sample._bitmasks,maskbits)
    # Calculate the number of masked stars at each pixel
    flaggedstars = np.sum(mask,axis=0)
    # Flag pixels where there aren't enough stars to do the fit
    flaggedpix = flaggedstars > (sample.numberStars()-minstar)
    mask.T[flaggedpix]=True
    return mask

class mask(subStarSample):
    """
    Define and apply a mask given a set of conditions in the form of the
    maskConditions function.
    
    """
    def __init__(self,dataSource,sampleType,maskFilter,ask=True,datadir='.',
                 func=None,badcombpixmask=4351,minSNR=50):
        """
        Mask a subsample according to a maskFilter function
        
        sampleType:   designator of the sample type - must be a key in readfn 
                      and independentVariables in data.py
        maskFilter:   function that decides on elements to be masked
        ask:          if True, function asks for user input to make 
                      filter_function.py, if False, uses existing 
                      filter_function.py
        
        """
        subStarSample.__init__(self,dataSource,sampleType,ask=ask,datadir=datadir,func=func)
        if isinstance(badcombpixmask,list):
            badcombpixmask=np.array(badcombpixmask)
        if isinstance(badcombpixmask,np.ndarray):
            if isinstance(badcombpixmask[0],str):
                badcombpixmask = 0
                for b in badcombpixmask:
                    badcombpixmask = np.sum(2**bitmask.apogee_pixmask_int(b))
            elif isinstance(badcombpixmask[0],int):
                badcombpixmask = np.sum(2**badcombpixmask)
        self.name+='/bm{0}'.format(badcombpixmask)
        self.getDirectory()
        self._SNR = self.spectra/self.spectra_errs
        self.minSNR = minSNR
        # find indices that should be masked
        self._maskHere = maskFilter(self,minstar=5,minSNR=self.minSNR,
                                    badcombpixmask=badcombpixmask)
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

