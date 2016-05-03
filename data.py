import apogee.tools.read as apread
from apogee.tools import bitmask
from read_clusterdata import read_caldata
import window as wn
import numpy as np

# Dictionary to translate APOGEE's pixel mask (DR12).
# Keys correspond to set bits in the mask.

aspcappix = 7214

APOGEE_PIXMASK={0:"BADPIX", # Pixel marked as BAD in bad pixel mask
                1:"CRPIX", # Pixel marked as cosmic ray in ap3d
                2:"SATPIX", # Pixel marked as saturated in ap3d
                3:"UNFIXABLE", # Pixel marked as unfixable in ap3d
                4:"BADDARK", # Pixel marked as bad as determined from dark frame
                5:"BADFLAT", # Pixel marked as bad as determined from flat frame
                6:"BADERR", # Pixel set to have very high error (not used)
                7:"NOSKY", # No sky available for this pixel from sky fibers
                8:"LITTROW_GHOST", # Pixel falls in Littrow ghost, may be affected
                9:"PERSIST_HIGH", # Pixel falls in high persistence region, may be affected
                10:"PERSIST_MED", # Pixel falls in medium persistence region, may be affected
                11:"PERSIST_LOW", # Pixel falls in low persistence region, may be affected
                12:"SIG_SKYLINE", # Pixel falls near sky line that has significant flux compared with object
                13:"SIG_TELLURIC", # Pixel falls near telluric line that has significant absorption
                14:"NOT_ENOUGH_PSF", # Less than 50 percent PSF in good pixels
                15:"POORSNR", # Signal to noise below limit
                16:"FAILFIT" # Fitting for stellar parameters failed on pixel
                } 

def timeIt(fn,*args,**kwargs):
    """
    A basic function to time how long another function takes to run.

    fn:     Function to time.

    Returns the output of the function and its runtime.

    """
    start = time.time()
    output = fn(*args,**kwargs)
    end = time.time()
    return output,end-start

# Chosen set of bits on which to mask
badcombpixmask= bitmask.badpixmask()+2**bitmask.apogee_pixmask_int("SIG_SKYLINE")

elems = ['Al','Ca','C','Fe','K','Mg','Mn','Na','Ni','N','O','Si','S','Ti','V']
elems = ['C','N','O','Na','Mg','Al','Si','S','K','Ca','Ti','V','Mn','Fe','Ni']


# Functions to access particular sample types
readfn = {'clusters' : read_caldata,        # Sample of open and globular clusters
          'OCs': read_caldata,                # Sample of open clusters
          'GCs': read_caldata,                # Sample of globular clusters
          'red_clump' : apread.rcsample        # Sample of red clump stars
          }

independentVariables = {'clusters':['TEFF'],
                        'OCs':['TEFF'],
                        'GCs':['TEFF'],
                        'red_clump':['TEFF','LOGG','FE_H']
                    }
elemwindows = {}
normwindows = np.zeros((len(elems),aspcappix))
e = 0
for elem in elems:
    w = wn.read(elem,dr=12,apStarWavegrid=False)
    nw = np.ma.masked_array(w/np.sqrt(np.sum(w)))
    elemwindows[elem] = w
    normwindows[e] = nw
    e+=1

#for testing
defaultparams={'red_clump':np.array([1.53796328e-07,3.80441208e-04,2.04021066e-07,
                                     -2.63714534e-08,-3.56518938e-08,-1.45798835e-06,
                                     -1.67953566e-07,-7.07997832e-09,1.92230060e-10,
                                     -1.23611443e-10]),
               'clusters':np.ones(3)
               }

coeff_inds = {'red_clump': ['const','Teff','logg','[Fe/H]','Teff^2','Teff*logg',
                            'Teff*[Fe/H]','logg^2','logg*[Fe/H]','[Fe/H]^2'],
              'clusters': ['const','Teff','Teff^2']
              }

detec_red_pix =[0,2920]
detec_blue_wv = [1.514,1.581] #microns
detec_green_pix = [2920,5320]
detec_green_wv = [1.585,1.644] #microns
detec_blue_pix = [5320,7213]
detec_red_wv = [1.647,1.696] #microns
detectors = [0,2920,5320,7214]
ASPCAPdetectors_pix = [(0,2920),(2920,5320),(5320,7214)]
apStarDetectors_pix = [(322,3242),(3648,6048),(6412,8306)]
detectors_wv = [(1.696,1.647),(1.644,1.585),(1.581,1.514)]

wv0 = 15100.802
totalpix = 8575
wvs = np.zeros(totalpix)
wvs[0] = wv0
for i in range(1,totalpix):
    wvs[i] = 10**(6e-6 + np.log10(wvs[i-1]))
aspcapwvs = np.concatenate((wvs[322:3242],wvs[3648:6048],wvs[6412:8306]))

    
def pixel2wavelength(pix,apStarWavegrid=False):
    if apStarWavegrid:
        if isinstance(pix,int):
            return wvs[pix]
        elif isinstance(pix,tuple):
            return wvs[pix[0]:pix[1]]
        elif isinstance(pix,(list,np.ndarray)):
            wavelengths = np.zeros(len(pix))
            for p in pix:
                wavelengths[p] = wvs[pix[p]]
    elif not apStarWavegrid:
        if isinstance(pix,int):
            return aspcapwvs[pix]
        elif isinstance(pix,tuple):
            return aspcapwvs[pix[0]:pix[1]]
        elif isinstance(pix,(list,np.ndarray)):
            wavelengths = np.zeros(len(pix))
            for p in pix:
                wavelengths[p] = aspcapwvs[pix[p]]

    
