import apogee.samples.rc as rcmodel
import apogee.tools.read as apread
from apogee.tools import bitmask
from read_clusterdata import read_caldata
import statsmodels.nonparametric.smoothers_lowess as sm
import scipy as sp
#import window as wn
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
    data= apread.allStar(main=True,exclude_star_bad=True,exclude_star_warn=True,rmdups=True)
    jk= data['J0']-data['K0']
    z= isodist.FEH2Z(data['METALS'],zsolar=0.017)
    z[z > 0.024]= 0.024
    logg= data['LOGG']
    indx= ((jk >= 0.8)#+(z < rcmodel.jkzcut(jk-0.1,upper=False))
            +(logg > rcmodel.loggteffcut(data['TEFF'],z,upper=True)))
    rgindx=indx*(data['METALS'] > -.8)
    return data[rgindx]

# Chosen set of bits on which to mask
badcombpixmask = bitmask.badpixmask()
badcombpixmask += 2**bitmask.apogee_pixmask_int("SIG_SKYLINE")

elems = ['C','N','O','Na','Mg','Al','Si','S','K','Ca','Ti','V','Mn','Fe','Ni']


# Functions to access particular sample types
readfn = {'clusters' : read_caldata,      # Sample of clusters
          'OCs': read_caldata,            # Sample of open clusters
          'GCs': read_caldata,            # Sample of globular clusters
          'red_clump' : apread.rcsample,  # Sample of red clump stars
          'red_giant' : rgsample          # Sample of red giant stars
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
                        'red_clump':['TEFF','LOGG','FE_H'],
                        'red_giant':['TEFF','LOGG','FE_H']
                    }

# Store information about where element windows are
# Track by element name
"""
elemwindows = {}
# Track by array in order of periodic table
normwindows = np.zeros((len(elems),aspcappix))
e = 0
for elem in elems:
    w = wn.read(elem,dr=13,apStarWavegrid=False)
    nw = np.ma.masked_array(w/np.sqrt(np.sum(w)))
    elemwindows[elem] = w
    normwindows[e] = nw
    e+=1
"""
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
                                              

def pix2wv(pix,apStarWavegrid=False):
    """
    NAME:
       pix2wv
    PURPOSE:
       convert pixel to wavelength
    INPUT:
       pix - pixel (int), range of pixels (tuple) or list of pixels (list/numpy array)
             float input will be converted to integers
       apStarWavegrid = (False) uses aspcapStarWavegrid by default
    OUTPUT:
       wavelength(s) in Angstroms corresponding to input pixel(s)
    HISTORY:
       2016-10-18 - Written - Price-Jones
    """
    # choose wavelength array to source from
    if apStarWavegrid:
        wvlist = wvs
    elif not apStarWavegrid:
        wvlist = aspcapwvs
    # Check input cases
    if isinstance(pix,float):
        pix = int(pix)
    if isinstance(pix,int):
        if pix > 0 and pix < totalpix:
            return wvlist[pix]
        else:
            print 
    elif isinstance(pix,tuple):
        if pix[0] > 0 and pix[1] < totalpix:
            return wvlist[int(pix[0]):int(pix[1]):int(pix[2])]
    elif isinstance(pix,(list,numpy.ndarray)):
        wavelengths = numpy.zeros(len(pix))
        for p in range(len(pix)):
            if pix[p] > 0 and pix[p] < totalpix:
                wavelengths[p] = wvlist[int(pix[p])]
        return wavelengths
    # If input not recognized inform the user
    elif not isinstance(wv,(int,float,tuple,list,numpy.ndarray)):
        print 'unrecognized pixel input'
        return None

def wv2pix(wv,apStarWavegrid=False):
    """
    NAME:
       wv2pix
    PURPOSE:
       convert wavelength to pixel using interpolated function
    INPUT:
       wv - wavelength (int), range of wavelengths (tuple) or list of wavelengths
            (list/numpy array) in Angstroms
       apStarWavegrid = (False) uses aspcapStarWavegrid by default
    OUTPUT:
       array of pixel(s) corresponding to input wavelength(s)
       nan - indicates input wavelength(s) outside the range
       None - indicates input wavelength type not recognized
       0 - indicates the wavelength can be found but is outside the bounds of the a
           spcapStarWavegrid
    HISTORY:
       2016-10-18 - Written - Price-Jones
    """
    # Check input cases
    if isinstance(wv,(int,float)):
        if wv >= wvs[-1] and wv <= wvs[0]:
            pixels = apStar_pixel_interp(wv)
        else:
            print 'wavelength outside wavelength range'
            return numpy.nan 
    elif isinstance(wv,tuple):
        if wv[0] >= wvs[-1] and wv[1] <= wvs[0]:
            wvlist = numpy.arange(wv[0],wv[1],wv[2])
            pixels = apStar_pixel_interp(wvlist)
        else:
            print 'range bounds lie outside wavelength range'
            return numpy.nan       
    elif isinstance(wv,(list,numpy.ndarray)):
        pixels = numpy.zeros(len(wv))
        allinside=True
        for w in range(len(wv)):
            if wv[w] >= wvs[-1] and wv[w] <= wvs[0]:
                pixels[w] = apStar_pixel_interp(wv[w])
            else:
                allinside=False
                pixels[w] = numpy.nan
        if not allinside:
            print 'some wavelengths outside wavelength range'
    # If input not recognized inform the user
    elif not isinstance(wv,(int,float,tuple,list,numpy.ndarray)):
        print 'unrecognized wavelength input'
        return None

    if apStarWavegrid:
        return pixels.astype(int)

    # If on aspcapStarWavegrid, convert appropriately    
    elif not apStarWavegrid:        
        # find where pixel list matches detectors
        blue = numpy.where((pixels >= apStarBlu_lo) & (pixels < apStarBlu_hi))
        green = numpy.where((pixels >= apStarGre_lo) & (pixels < apStarGre_hi))
        red = numpy.where((pixels >= apStarRed_lo) & (pixels < apStarRed_hi))
        # find where pixel list does not match detectors
        if pixels.size > 1:
            nomatch = (numpy.array([i for i in range(len(pixels)) if i not in blue[0] and i not in green[0] and i not in red[0]]),)
             # adjust pixel values to match aspcap wavegrid
            pixels[blue] -= (apStarBlu_lo-aspcapBlu_start)
            pixels[green] -= (apStarGre_lo-aspcapGre_start)
            pixels[red] -= (apStarRed_lo-aspcapRed_start)
        # Case of single wavelength
        elif pixels.size == 1:
            if blue[0].size==1:
                pixels -= (apStarBlu_lo-aspcapBlu_start)
            elif green[0].size==1:
                pixels -= (apStarGre_lo-aspcapGre_start)
            elif red[0].size==1:
                pixels -= (apStarRed_lo-aspcapRed_start)
            elif blue[0].size==0 and green[0].size==0 and red[0].size==0:
                nomatch = ([0],)
                pixels = 0
        return numpy.floor(pixels).astype(int)
