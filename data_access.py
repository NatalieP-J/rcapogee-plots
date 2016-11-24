import apogee.samples.rc as rcmodel
import apogee.tools.read as apread
from apogee.tools import bitmask
from read_clusterdata import read_caldata
import statsmodels.nonparametric.smoothers_lowess as sm
import scipy as sp
import numpy as np
import isodist
import os
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

def rgsample(dr='13'):
    data= apread.allStar(main=True,exclude_star_bad=True,exclude_star_warn=True,rmdups=True,dr=dr)
    jk= data['J0']-data['K0']
    z= isodist.FEH2Z(data['METALS'],zsolar=0.017)
    z[z > 0.024]= 0.024
    logg= data['LOGG']
    indx= ((jk >= 0.8)#+(z < rcmodel.jkzcut(jk-0.1,upper=False))
            +(logg > rcmodel.loggteffcut(data['TEFF'],z,upper=True)))
    rgindx=indx*(data['METALS'] > -.8)
    return data[rgindx]


# Functions to access particular sample types
readfn = {'apogee':{'clusters' : read_caldata,      # Sample of clusters
          'OCs': read_caldata,            # Sample of open clusters
          'GCs': read_caldata,            # Sample of globular clusters
          'red_clump' : apread.rcsample,  # Sample of red clump stars
          'red_giant' : rgsample}         # Sample of red giant stars
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
independentVariables = {'apogee':{'clusters':['TEFF'],
                        'OCs':['TEFF'],
                        'GCs':['TEFF'],
                        'red_clump':['TEFF','LOGG','FE_H'],
                        'red_giant':['TEFF','LOGG','FE_H']}
                    }


# Pixels at which detectors transition for each wavegrid
ASPCAPdetectors_pix = [(0,2920),(2920,5320),(5320,7214)]
apStarDetectors_pix = [(322,3242),(3648,6048),(6412,8306)]

# Assign list used in smoothing function
if aspcappix==7214:
    detectors = [0,2920,5320,7214]

# Expected wavelengths at each detector
detectors_wv = [(1.696,1.647),(1.644,1.585),(1.581,1.514)]

wvpath = 'apogee_wavelengths'
totalpix = 8575 # total pixels on the apstar grid
if os.path.isfile(wvpath):
    wvs = np.loadtxt(wvpath)
elif not os.path.isfile(wvpath):
    # Make array of wavelengths for the full apStar wavegrid,
    # then concatenate a version for the aspcap wavegrid
    wv0 = 10**4.179 # starting wavelength
    wvs = np.zeros(totalpix) # create empty array for wavelengths
    # Assign all wavelengths
    wvs[0] = wv0
    for i in range(1,totalpix):
        wvs[i] = 10**(6e-6 + np.log10(wvs[i-1]))
    wvs = wvs[::-1]
    np.savetxt(wvpath,wvs)
pixels = np.arange(0,totalpix)
aspcapwvs = np.concatenate((wvs[322:3242],wvs[3648:6048],wvs[6412:8306]))
apStar_pixel_interp = sp.interpolate.interp1d(wvs,pixels,kind='linear',
                                              bounds_error=False)
apStar_wv_interp = sp.interpolate.interp1d(pixels,wvs,kind='linear',
                                           bounds_error=False)
                                              

def pixel2wavelength(pix,apStarWavegrid=False):
    """
    Convert pixel to wavelength.

    pix:              Pixel (int), range of pixels (tuple) or list of pixels 
                      (list/array) to be converted. Non integer inputs will 
                      use an interpolated solution for wavelengths.
    apStarWavegrid:   Specifies whether to use the apStar wavegrid (True) or 
                      the aspcap one (False)

    Returns the wavelength(s) (in Angstroms) corresponding to input pixels
    """
    if apStarWavegrid:
        # Use apStar wavegrid
        if isinstance(pix,float):
            pix = int(pix)
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
        if isinstance(pix,float):
            pix = int(pix)
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
    
    wv:               wavelength, range of wavelengths (tuple with form 
                      (lower limit, upper limit, stepsize)) or list of 
                      wavelengths - UNITS angstroms
                      
    apStarWavegrid:   specifies whether to use the apStar wavegrid (True) or 
                      the aspcap one (False)

    Returns the pixel that contains the input wavelength(s).
    """
    if apStarWavegrid:
        # use apStar wavegrid

        # if single value, return single value
        if isinstance(wv,(int,float)):
            if wv >= wvs[-1] and wv <= wvs[0]:
                return int(np.floor(apStar_pixel_interp(wv)))
            else:
                print 'wavelength outside wavelength range, returning nan'
                return np.nan 
        # if tuple, return array of pixels corresponding to the defined 
        # wavelength array
        elif isinstance(wv,tuple):
            if wv[0] < wvs[-1] or wv[1] > wvs[0]:
                print 'range bounds lie outside wavelength range, returning nan'
                return np.nan
            else:
                wvlist = np.arange(wv[0],wv[1],wv[2])
                return np.floor(apStar_pixel_interp(wvlist)).astype(int)
       # if list, return array of pixels corresponding to the list
        elif isinstance(wv,(list,np.ndarray)):
            pixels = np.zeros(len(wv))
            for w in range(len(wv)):
                print wvs[w]
                if wv[w] >= wvs[-1] and wv[w] <= wvs[0]:
                    pixels[w] = apStar_pixel_interp(wv[w])
                else:
                    pixels[w] = np.nan
                print pixels[w]
            return np.floor(pixels).astype(int)
    
    elif not apStarWavegrid:
        # use aspcap wavegrid
        
        # if single value, define single value
        if isinstance(wv,(int,float)):
            if wv >= wvs[-1] and wv <= wvs[0]:
                pixels = np.array([apStar_pixel_interp(wv)])
            else:
                print 'wavelength outside wavelength range, returning nan'
                return np.nan 
        # if tuple, define array of pixels corresponding to the defined 
        # wavelength array
        elif isinstance(wv,tuple):
            if wv[0] < wvs[-1] or wv[1] > wvs[0]:
                print 'range bounds lie outside wavelength range, returning nan'
                return np.nan
            else:
                wvlist = np.arange(wv[0],wv[1],wv[2])
                pixels = np.array([apStar_pixel_interp(wvlist)])
        # if list, return array of pixels corresponding to the list
        elif isinstance(wv,(list,np.ndarray)):
            pixels = np.zeros(len(wv))
            for w in range(len(wv)):
                if wv[w] >= wvs[-1] and wv[w] <= wvs[0]:
                    pixels[w] = apStar_pixel_interp(wv[w])
                    print pixels[w]
                else:
                    pixels[w] = np.nan
        
        # find where pixel list matches detectors
        blue = np.where((pixels >= 322) & (pixels < 3242))
        green = np.where((pixels >= 3648) & (pixels < 6048))
        red = np.where((pixels >= 6412) & (pixels < 8306))
        # find where pixel list does not match detectors
        nomatch = (np.array([i for i in range(len(pixels)) if i not in blue and i not in green and i not in red]),)
        # adjust pixel values to match aspcap wavegrid
        pixels[blue] -= 322
        pixels[green] -= (3648-2920)
        pixels[red] -= (6412-5320)
        if nomatch[0].size != 0:
            pixels[nomatch] = np.nan
        return np.floor(pixels).astype(int)
