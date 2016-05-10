import apogee.tools.read as apread
from apogee.tools import bitmask
from read_clusterdata import read_caldata
import statsmodels.nonparametric.smoothers_lowess as sm
import scipy as sp
import window as wn
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  20
}

matplotlib.rc('font',**font)
plt.ion()



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

def smoothMedian(diag,frac=None,numpix=None):
    """
    Uses Locally Weighted Scatterplot Smoothing to smooth an array on 
    each detector separately. Interpolates at masked pixels and concatenates 
    the result.
    
    Returns the smoothed median value of the input array, with the same 
    dimension.
    """
    mask = diag.mask==False
    smoothmedian = np.zeros(diag.shape)
    for i in range(len(detectors)-1):
        xarray = np.arange(detectors[i],detectors[i+1])
        yarray = diag[detectors[i]:detectors[i+1]]
        array_mask = mask[detectors[i]:detectors[i+1]]
        if frac:
            low_smooth = sm.lowess(yarray[array_mask],xarray[array_mask],
                                   frac=frac,it=3,return_sorted=False)
        if numpix:
            frac = float(numpix)/len(xarray)
            low_smooth = sm.lowess(yarray[array_mask],xarray[array_mask],
                                   frac=frac,it=3,return_sorted=False)
        smooth_interp = sp.interpolate.interp1d(xarray[array_mask],
                                                low_smooth,bounds_error=False)
        smoothmedian[detectors[i]:detectors[i+1]] = smooth_interp(xarray)
    nanlocs = np.where(np.isnan(smoothmedian))
    smoothmedian[nanlocs] = 1
    return smoothmedian

# Chosen set of bits on which to mask
badcombpixmask = bitmask.badpixmask()
badcombpixmask += 2**bitmask.apogee_pixmask_int("SIG_SKYLINE")

elems = ['C','N','O','Na','Mg','Al','Si','S','K','Ca','Ti','V','Mn','Fe','Ni']


# Functions to access particular sample types
readfn = {'clusters' : read_caldata,      # Sample of clusters
          'OCs': read_caldata,            # Sample of open clusters
          'GCs': read_caldata,            # Sample of globular clusters
          'red_clump' : apread.rcsample   # Sample of red clump stars
          }

# List of accepted keys to do slice in
keyList = ['RA','DEC','GLON','GLAT','TEFF','LOGG','TEFF_ERR','LOGG_ERR',
            'AL_H','CA_H','C_H','FE_H','K_H','MG_H','MN_H','NA_H','NI_H',
            'N_H','O_H','SI_H','S_H','TI_H','V_H','CLUSTER']
keyList.sort()

# List of accepted keys for upper and lower limits
_upperKeys = ['max','m','Max','Maximum','maximum','']
_lowerKeys = ['min','m','Min','Minimum','minimum','']

# Specify which independent variables to use when fitting different sample types
independentVariables = {'clusters':['TEFF'],
                        'OCs':['TEFF'],
                        'GCs':['TEFF'],
                        'red_clump':['TEFF','LOGG','FE_H']
                    }

# Store information about where element windows are
# Track by element name
elemwindows = {}
# Track by array in order of periodic table
normwindows = np.zeros((len(elems),aspcappix))
e = 0
for elem in elems:
    w = wn.read(elem,dr=12,apStarWavegrid=False)
    nw = np.ma.masked_array(w/np.sqrt(np.sum(w)))
    elemwindows[elem] = w
    normwindows[e] = nw
    e+=1

def pixel2element(arr):
    """
    Convert an array in pixel space to a corresponding array in pseudo-element 
    space.

    arr:   array to convert - must have one dimension equal to aspcappix

    Returns reshaped array.
    """
    # Determine which direction the array matches with pixel space and dot it 
    # with normalized element windows.
    if arr.shape[1] == aspcappix:
        return np.dot(arr,normwindows.T)
    elif arr.shape[0] == aspcappix:
        return np.dot(arr.T,normwindow.T)

# Test parameters to use for testing the polynomial fit
defaultparams={'red_clump':np.array([1.53796328e-07,3.80441208e-04,
                                     2.04021066e-07,-2.63714534e-08,
                                     -3.56518938e-08,-1.45798835e-06,
                                     -1.67953566e-07,-7.07997832e-09,
                                     1.92230060e-10,-1.23611443e-10]),
               'clusters':np.ones(3)
               }

# Track what each coefficient corresponds to in polynomial fits
coeff_inds = {'red_clump': ['const','Teff','logg','[Fe/H]','Teff^2','Teff*logg',
                            'Teff*[Fe/H]','logg^2','logg*[Fe/H]','[Fe/H]^2'],
              'clusters': ['const','Teff','Teff^2']
              }


# Pixels at which detectors transition for each wavegrid
ASPCAPdetectors_pix = [(0,2920),(2920,5320),(5320,7214)]
apStarDetectors_pix = [(322,3242),(3648,6048),(6412,8306)]

# Assign list used in smoothing function
if aspcappix==7214:
    detectors = [0,2920,5320,7214]

# Expected wavelengths at each detector
detectors_wv = [(1.696,1.647),(1.644,1.585),(1.581,1.514)]

# Make array of wavelengths for the full apStar wavegrid,
# then concatenate a version for the aspcap wavegrid
wv0 = 10**4.179 # starting wavelength
totalpix = 8575 # total pixels on the apstar grid
wvs = np.zeros(totalpix) # create empty array fo wavelengths
# Assign all wavelengths
wvs[0] = wv0
for i in range(1,totalpix):
    wvs[i] = 10**(6e-6 + np.log10(wvs[i-1]))
aspcapwvs = np.concatenate((wvs[322:3242],wvs[3648:6048],wvs[6412:8306]))

apStar_pixel_interp = sp.interpolate.interp1d(wvs,np.arange(0,totalpix),
                                              kind='linear',bounds_error=False)
                                              

def pixel2wavelength(pix,apStarWavegrid=False):
    """
    Convert pixel to wavelength.

    pix:              pixel (int), range of pixels (tuple) or list of pixels 
                      (list/array) to be converted
    apStarWavegrid:   specifies whether to use the apStar wavegrid (True) or 
                      the aspcap one (False)
    """
    if apStarWavegrid:
        # Use apStar wavegrid
        if isinstance(pix,int):
            return wvs[pix]
        elif isinstance(pix,tuple):
            return wvs[pix[0]:pix[1]]
        elif isinstance(pix,(list,np.ndarray)):
            wavelengths = np.zeros(len(pix))
            for p in pix:
                wavelengths[p] = wvs[pix[p]]
            return wavelengths
    elif not apStarWavegrid:
        # Use aspcap wavegrid
        if isinstance(pix,int):
            return aspcapwvs[pix]
        elif isinstance(pix,tuple):
            return aspcapwvs[pix[0]:pix[1]]
        elif isinstance(pix,(list,np.ndarray)):
            wavelengths = np.zeros(len(pix))
            for p in pix:
                wavelengths[p] = aspcapwvs[pix[p]]
            return wavelengths

def wavelength2pixel(wv,apStarWavegrid=False):
    """
    Convert wavelength to pixel.
    
    pix:              pixel (int), range of pixels (tuple with form (lower 
                      limit, upper limit, stepsize)) or list of pixels 
                      (list/array) to be converted
    apStarWavegrid:   specifies whether to use the apStar wavegrid (True) or 
                      the aspcap one (False)
    
    """
    if apStarWavegrid:
        # use apStar wavegrid

        # if single value, return single value
        if isinstance(wv,(int,float)):
            if wv >= wvs[0] and wv <= wvs[-1]:
                return int(np.floor(apStar_pixel_interp(wv)))
            else:
                print 'wavelength outside wavelength range, returning nan'
                return np.nan 
        # if tuple, return array of pixels corresponding to the defined 
        # wavelength array
        elif isinstance(wv,tuple):
            if wv[0] < wvs[0] or wv[1] > wvs[-1]:
                print 'range bounds lie outside wavelength range, returning nan'
                return np.nan
            else:
                wvlist = np.arange(wv[0],wv[1],wv[2])
                return np.floor(apStar_pixel_interp(wvlist)).astype(int)
       # if list, return array of pixels corresponding to the list
        elif isinstance(wv,(list,np.ndarray)):
            pixels = np.zeros(len(wv))
            for w in range(len(wv)):
                if w >= wvs[0] and wv <= wvs[-1]:
                    pixels[w] = apStar_pixel_interp(wv[w])
                else:
                    pixels[w] = np.nan
            return np.floor(pixels).astype(int)
    
    elif not apStarWavegrid:
        # use aspcap wavegrid
        
        # if single value, define single value
        if isinstance(wv,(int,float)):
            if wv >= wvs[0] and wv <= wvs[-1]:
                pixels = np.array([apStar_pixel_interp(wv)])
            else:
                print 'wavelength outside wavelength range, returning nan'
                return np.nan 
        # if tuple, define array of pixels corresponding to the defined 
        # wavelength array
        elif isinstance(wv,tuple):
            if wv[0] < wvs[0] or wv[1] > wvs[-1]:
                print 'range bounds lie outside wavelength range, returning nan'
                return np.nan
            else:
                wvlist = np.arange(wv[0],wv[1],wv[2])
                pixels = np.array([apStar_pixel_interp(wvlist)])
        # if list, return array of pixels corresponding to the list
        elif isinstance(wv,(list,np.ndarray)):
            pixels = np.zeros(len(wv))
            for w in range(len(wv)):
                if w >= wvs[0] and wv <= wvs[-1]:
                    pixels[w] = apStar_pixel_interp(wv[w])
                else:
                    pixels[w] = np.nan
        
        # find where pixel list matches detectors
        blue = np.where((pixels >= 322) & (pixels < 3242))
        green = np.where((pixels >= 3648) & (pixels < 6048))
        red = np.where((pixels >= 6412) & (pixels < 8306))
        # find where pixel list does not match detectors
        none = (np.array([i for i in range(len(pixels)) if i not in blue and i not in green and i not in red]),)
        # adjust pixel values to match aspcap wavegrid
        pixels[blue] -= 322
        pixels[green] -= (3648-2920)
        pixels[red] -= (6412-5320)
        pixels[none] = np.nan
        return np.floor(pixels).astype(int)
