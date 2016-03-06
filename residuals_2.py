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

_keyList = ['RA','DEC','GLON','GLAT','TEFF','LOGG','TEFF_ERR','LOGG_ERR',
            'AL_H','CA_H','C_H','FE_H','K_H','MG_H','MN_H','NA_H','NI_H',
            'N_H','O_H','SI_H','S_H','TI_H','V_H']
_keyList.sort()

_upperKeys = ['max','m','Max','Maximum','maximum','']
_lowerKeys = ['min','m','Min','Minimum','minimum','']

def basicIronFilter(data):
    """
    A sample filter function.
    """
    return (data['FE_H'] > -0.15) & (data['FE_H'] < -0.1)

def bitsNotSet(bitmask,maskbits):
    """
    Given a bitmask, returns False where any of maskbits are set and True otherwise.
    """
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

    def makeArrays(self,data):
        """
        Create arrays across all stars in the sample with shape number of stars by aspcappix.
        
        """

        # Create fit variable arrays
        self.teff = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.logg = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.fe_h = np.ma.masked_array(np.zeros((len(data),aspcappix)))

        # Create fit variable uncertainty arrays
        self.teff_err = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.logg_err = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.fe_h_err = np.ma.masked_array(np.zeros((len(data),aspcappix)))

        # Create spectra arrays
        self.spectra = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self.spectra_errs = np.ma.masked_array(np.zeros((len(data),aspcappix)))
        self._bitmasks = np.ma.masked_array(np.zeros((len(data),aspcappix)))

        # Fill arrays for each star
        for star in range(len(data)):
            LOC = data[star]['LOCATION_ID']
            APO = data[star]['APOGEE_ID']
            TEFF = data[star]['TEFF']
            LOGG = data[star]['LOGG']
            FE_H = data[star]['FE_H']
            TEFF_ERR = data[star]['TEFF_ERR']
            LOGG_ERR = data[star]['LOGG_ERR']
            FE_H_ERR = data[star]['FE_H_ERR']
            # Fit variables
            self.teff[star] = np.ma.masked_array([TEFF]*aspcappix)
            self.logg[star] = np.ma.masked_array([LOGG]*aspcappix)
            self.fe_h[star] = np.ma.masked_array([FE_H]*aspcappix)
            
            # Fit variable uncertainty
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

            
            
class makeFilter(starSample):
    """
    Contains functions to create a filter and associated name for a starSample.
    """
    def __init__(self,sampleType,ask=True):
        starSample.__init__(self,sampleType)
        if ask:
            self.done = False
            print 'Type done at any prompt when finished'
            self.name = self._sampleType
            self.condition = ''
            while not self.done:
                self._sampleInfo()
            if self.condition == '':
                print 'No conditions set'
                self.__init__(sampleType,ask=True)
            self.condition = self.condition[:-2]
            f = open('filter_function.py','w')
            f.write(self._basicStructure()+self.condition)
            f.close()
        elif not ask:
            try:
                import filter_function
                reload(filter_function)
                from filter_function import starFilter
                name = starFilter.__doc__.split('\n')[-2]
            except ImportError:
                print 'filter_function.py does not contain the required starFilter function.'
                self.__init__(sampleType,ask=True)
            
    def _basicStructure(self):
        return 'def starFilter(data):\n\t"""\n\t{0}\n\t"""\n\treturn'.format(self.name)

    def _sampleInfo(self):
        while not self.done:
            gotKey = False
            while not gotKey:
                key = raw_input('Data key: ')
                if key in _keyList:
                    self.name+='_'+key
                    match = self._match(key)
                    print 'Match', match
                    if match=='done':
                        self.done = True
                        break
                    elif match[0]==True:
                        self.name+='_match'+match[1]
                        self.condition += ' (data[\'{0}\'] == {1}) &'.format(key,match[1])
                    elif match[0]==False:
                        self.name+='_up'+str(match[1])+'_lo'+str(match[2])
                        self.condition += ' (data[\'{0}\'] < {1}) & (data[\'{0}\'] > {2}) &'.format(key,match[1],match[2])
                elif key not in _keyList and key != 'done':
                    print 'Got a bad key. Try choosing one of ',_keyList
                    self._sampleInfo()
                elif key == 'done':
                    print 'Done getting filter information'
                    self.done = True
                    break
    

    def _match(self,key):
        match = raw_input('Default is full range. Match or slice? ').strip()
        if match == 'match' or match == 'm' or match == 'Match':
            m = raw_input('Match value: ')
            if m=='done':
                print 'Done getting filter information'
                return 'done'
            elif m!='done' and m in self.data[key]:
                return True,m
            elif m not in self.data[key]:
                print 'No match for this key. Try choosing one of ',np.unique(self.data[key])
                self._match(key)
        elif match == 'slice' or match == 's' or match == 'Slice':
            upperLimit = raw_input('Upper limit (Enter for maximum): ')
            lowerLimit = raw_input('Lower limit (Enter for minimum): ')
            if upperLimit <= lowerLimit:
                print 'Limits are the same or are in the wrong order. Try again.'
                self._match(key)
            elif upperLimit == 'done' or lowerLimit == 'done':
                print 'Done getting filter information'
                return 'done'
            elif upperLimit != 'done' and lowerLimit != 'done':
                if upperLimit == 'max' or upperLimit == 'm' or upperLimit == '':
                    upperLimit = np.max(self.data[key])
                if lowerLimit == 'min' or lowerLimit == 'm' or lowerLimit == '':
                    lowerLimit = np.min(self.data[key])
                try:
                    print 'Found good limits'
                    return False,float(upperLimit),float(lowerLimit)
                except ValueError as e:
                    print 'Please enter floats for the limits'
                    self._match(key)
        elif match == 'done':
            print 'Done getting filter information'
            return 'done'

        else:
            print 'Please type m or s'
            self._match(key)
        
    def getDirectory(self):
        """
        Create directory for given filter
        """
        return self.name




class subStarSample(makeFilter):
    def __init__(self,sampleType,ask=True):
        """
        Create a subsample according to a starFilter function
        Set ask=True if you wish to be asked about the filter, leave False to use existing filter

        """
        makeFilter.__init__(self,sampleType,ask=ask)
        import filter_function
        reload(filter_function)
        from filter_function import starFilter
        self._matchingStars = starFilter(self.data)
        print len(self.data),' total stars'
        self.matchingData = self.data[self._matchingStars]
        print len(self.matchingData),' stars'
        self.makeArrays(self.matchingData)        


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
