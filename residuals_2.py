import numpy as np

from data import *

# Import APOGEE/sample packages
from apogee.tools import bitmask
import window as wn
from read_clusterdata import read_caldata
import apogee.spec.plot as aplt

# Import fitting and analysis packages
import access_spectrum
reload(access_spectrum)
import access_spectrum as acs
import reduce_dataset as rd
import polyfit as pf

# Example star filter function
def starFilter(data):
    """
    Returns True where stellar properties match conditions
    
    """
    return (data['TEFF'] > 4500) & (data['TEFF'] < 4800)

def basicIronFilter(data):
    return (data['FE_H'] > -0.15) & (data['FE_H'] < -0.1)

def bitsNotSet(bitmask,maskbits):
    goodLocs_bool = np.ones(bitmask.shape).astype(bool)
    for m in maskbits:
        bitind = bitmask.bit_set(m,bitmask)
        goodLocs_bool[bitind] = False
    return goodLocs_bool

# Example mask filter function
def maskFilter(sample):
    """
    Returns True where sample properties match conditions

    """
    maskbits = bitmask.bits_set(badcombpixmask)
    return (sample._SNR > 50.) & (sample._SNR < 200.) & bitsNotSet(sample._bitmasks,maskbits)

class starInfo(object):
    """
    Each instance of starInfo holds information about a given star
    """
    def __init__(self,data):
        """
        For a given data index, assign stellar properties.

        """
        self.LOC = data['LOCATION_ID']
        self.APO = data['APOGEE_ID']
        self.TEFF = data['TEFF']
        self.LOGG = data['LOGG']
        self.FE_H = data['FE_H']
        self.TEFF_ERR = data['TEFF_ERR']
        self.LOGG_ERR = data['LOGG_ERR']
        self.FE_H_ERR = data['FE_H_ERR']
        
    def getData(self):
        """
        Create arrays of data for this star.

        """
        # Independent variables
        self._teff = np.ma.masked_array([self.TEFF]*aspcappix)
        self._logg = np.ma.masked_array([self.LOGG]*aspcappix)
        self._fe_h = np.ma.masked_array([self.FE_H]*aspcappix)
        
        # Independent variable uncertainty
        self._teff_err = np.ma.masked_array([self.TEFF_ERR]*aspcappix)
        self._logg_err = np.ma.masked_array([self.LOGG_ERR]*aspcappix)
        self._fe_h_err = np.ma.masked_array([self.FE_H_ERR]*aspcappix)
        
        # Spectral data
        self.spectrum = apread.aspcapStar(self.LOC,self.APO,ext=1,header=False, 
                                          aspcapWavegrid=True)
        self.spectrum_err = apread.aspcapStar(self.LOC,self.APO,ext=2,header=False, 
                                              aspcapWavegrid=True)
        self._bitmask = apread.apStar(self.LOC,self.APO,
                                     ext=3, header=False, aspcapWavegrid=True)[1]

class starSample(object):
    def __init__(self,sampleType):
        """
        Get properties for all stars that match the sample type
        """
        self._sampleType = sampleType
        self._getProperties()

    def _getProperties(self):
        """
        Get properties of all possible stars to be used.
        """
        self.data = readfn[self._sampleType]()

    def getStars(self,data):
        """
        Create a starInfo object for each star in data.

        """
        self.allStars = []
        self.numberStars = len(data)
        for star in range(len(data)):
            newStar = starInfo(data[star])
            newStar.getData()
            self.allStars.append(newStar)

    def makeArrays(self,data):
        """
        Create arrays across all stars in the sample with shape number of stars by aspcappix.
        
        """

        self.teff = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.logg = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.fe_h = np.ma.masked_array(np.zeros((len(data),aspcappix)))

        self.teff_err = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.logg_err = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.fe_h_err = np.ma.masked_array(np.zeros((len(data),aspcappix)))

        self.spectra = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.spectra_errs = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self._bitmasks = np.ma.masked_array(np.zeros((len(data),aspcappix)))

        for star in range(len(data)):
            LOC = data[star]['LOCATION_ID']
            APO = data[star]['APOGEE_ID']
            TEFF = data[star]['TEFF']
            LOGG = data[star]['LOGG']
            FE_H = data[star]['FE_H']
            TEFF_ERR = data[star]['TEFF_ERR']
            LOGG_ERR = data[star]['LOGG_ERR']
            FE_H_ERR = data[star]['FE_H_ERR']
            self.teff[star] = np.ma.masked_array([TEFF]*aspcappix)
            self.logg[star] = np.ma.masked_array([LOGG]*aspcappix)
            self.fe_h[star] = np.ma.masked_array([FE_H]*aspcappix)
            
            # Independent variable uncertainty
            self.teff_err[star] = np.ma.masked_array([TEFF_ERR]*aspcappix)
            self.logg_err[star] = np.ma.masked_array([LOGG_ERR]*aspcappix)
            self.fe_h_err[star] = np.ma.masked_array([FE_H_ERR]*aspcappix)
            
            # Spectral data
            self.spectra[star] = apread.aspcapStar(LOC,APO,ext=1,header=False, 
                                                   aspcapWavegrid=True)
            self.spectra_errs[star] = apread.aspcapStar(LOC,APO,ext=2,header=False, 
                                            aspcapWavegrid=True)
            self._bitmasks[star] = apread.apStar(LOC,APO,ext=3, header=False, 
                                    aspcapWavegrid=True)[1]            

            
            

class subStarSample(starSample):
    def __init__(self,sampleType,starFilter):
        """
        Create a subsample according to a starFilter function
        starSample call:

        subStarSample(red_clump,lambda data: data['TEFF'] > 4500)
        subStarSample(red_clump,starFilter)

        """
        starSample.__init__(self,sampleType)
        self._matchingStars = starFilter(self.data)
        self.matchingData = self.data[self._matchingStars]
        print 'Found matching stars'
        print 'Number of stars: ',len(self.matchingData)
        self.getStars(self.matchingData)
        print 'Got stars'
        self.makeArrays()        


class mask(subStarSample):
    """
    Define a mask given a conditions in the form of maskConditions
    """
    def __init__(self,sampleType,starFilter,maskConditions):
        subStarSample.__init__(self,sampleType,starFilter)
        self._SNR = self.spectra/self.spectra_errs
        self.masked = np.zeros(self.spectra.shape)
        self.unmasked = np.ones(self.spectra.shape)
        self._maskHere = maskConditions(self)
        self.masked[_maskHere]= 1
        self.unmasked[_maskHere] = 0
        self.applyMask()

    def applyMask(self):
        """
        Mask all arrays according to maskConditions

        """
        self.teff.mask = self.masked
        self.logg.mask = self.masked
        self.fe_h.mask = self.masked

        self.teff_err.mask = self.masked
        self.logg_err.mask = self.masked
        self.fe_h_err.mask = self.masked

        self.spectra.mask = self.masked
        self.spectra_errs.mask = self.masked


#class fit(mask)

#class residuals(fit)

#class uncertainty(fit)
