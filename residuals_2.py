import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import multiprocessing
import matplotlib.pyplot as plt
from data import *

# Import APOGEE/sample packages
from apogee.tools import bitmask as bm
from read_clusterdata import read_caldata
import apogee.spec.plot as aplt

# Import fitting and analysis packages
import access_spectrum
reload(access_spectrum)
import access_spectrum as acs
import reduce_dataset as rd
import polyfit as pf

#Import EMPCA package
from empca import empca

# List of accepted keys to do slice in
_keyList = ['RA','DEC','GLON','GLAT','TEFF','LOGG','TEFF_ERR','LOGG_ERR',
            'AL_H','CA_H','C_H','FE_H','K_H','MG_H','MN_H','NA_H','NI_H',
            'N_H','O_H','SI_H','S_H','TI_H','V_H','CLUSTER']
_keyList.sort()

# List of accepted keys for upper and lower limits
_upperKeys = ['max','m','Max','Maximum','maximum','']
_lowerKeys = ['min','m','Min','Minimum','minimum','']

#def directoryClean():
    

def bitsNotSet(bitmask,maskbits):
    """
    Given a bitmask, returns True where any of maskbits are set and False otherwise.
    """
    goodLocs_bool = np.zeros(bitmask.shape).astype(bool)
    for m in maskbits:
        bitind = bm.bit_set(m,bitmask)
        goodLocs_bool[bitind] = True
    return goodLocs_bool

# Example mask filter function
def maskFilter(sample):
    """
    Returns True where sample properties match conditions

    """
    maskbits = bm.bits_set(badcombpixmask)
    return (sample._SNR < 50.) | (sample._SNR > 200.) | bitsNotSet(sample._bitmasks,maskbits)

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
        self.teff = np.ma.masked_array(np.zeros((len(data),aspcappix),dtype=float))
        self.logg = np.ma.masked_array(np.zeros((len(data),aspcappix),dtype=float))
        self.fe_h = np.ma.masked_array(np.zeros((len(data),aspcappix),dtype=float))

        # Create spectra arrays
        self.spectra = np.ma.masked_array(np.zeros((len(data),aspcappix),dtype=float))
        self.spectra_errs = np.ma.masked_array(np.zeros((len(data),aspcappix),dtype=float))
        self._bitmasks = np.zeros((len(data),aspcappix),dtype=np.int64)

        # Fill arrays for each star
        for star in range(len(data)):
            LOC = data[star]['LOCATION_ID']
            APO = data[star]['APOGEE_ID']
            TEFF = data[star]['TEFF']
            LOGG = data[star]['LOGG']
            FE_H = data[star]['FE_H']

            # Fit variables
            self.teff[star] = np.ma.masked_array([TEFF]*aspcappix)
            self.logg[star] = np.ma.masked_array([LOGG]*aspcappix)
            self.fe_h[star] = np.ma.masked_array([FE_H]*aspcappix)
            
            # Spectral data
            self.spectra[star] = apread.aspcapStar(LOC,APO,ext=1,header=False, 
                                                   aspcapWavegrid=True)
            self.spectra_errs[star] = apread.aspcapStar(LOC,APO,ext=2,header=False, 
                                                        aspcapWavegrid=True)
            self._bitmasks[star] = apread.apStar(LOC,APO,ext=3, header=False, 
                                                 aspcapWavegrid=True)[1] 
    def plotHistogram(self,array,title = '',xlabel = '',ylabel = 'number of stars',saveName=None,**kwargs):
        hist,binEdges = np.histogram(array,**kwargs)
        area = np.sum(hist*(binEdges[1]-binEdges[0]))
        plt.bar(binEdges[:-1],hist/area,width = binEdges[1]-binEdges[0])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(0,1)
        if saveName:
            plt.savefig('plots/'+saveName+'.png')
            plt.close()
            
            
class makeFilter(starSample):
    """
    Contains functions to create a filter and associated name for a starSample.
    """
    def __init__(self,sampleType,ask=True):
        """
        Sets up filter_function.py file to contain the appropriate function and puts the save directory name in 
        the docstring of the function.

        sampleType:   designator of the sample type - must be a key in readfn and independentVariables in data.py
        ask:          if True, function asks for user input to make filter_function.py, if False, uses existing
                      filter_function.py
                      
        """
        starSample.__init__(self,sampleType)
        if ask:
            self.done = False
            print 'Type done at any prompt when finished'
            # Start name and condition string
            self.name = self._sampleType
            self.condition = ''
            # Ask for new key conditions until the user signals done
            while not self.done:
                self._sampleInfo()
            # Check that the user set conditions. If conditions not set, recursively call init
            if self.condition == '':
                print 'No conditions set'
                self.__init__(sampleType,ask=True)
            # When conditions set, trim trailing ampersand
            self.condition = self.condition[:-2]
            # Write the new filter function to file
            f = open('filter_function.py','w')
            f.write(self._basicStructure()+self.condition)
            f.close()
        elif not ask:
            # Import existing filter function. If function doesn't exist, recursively call init
            try:
                import filter_function
                reload(filter_function)
                from filter_function import starFilter
                self.name = starFilter.__doc__.split('\n')[-2]
                self.name = self.name.split('\t')[-1]
            except ImportError:
                print 'filter_function.py does not contain the required starFilter function.'
                self.__init__(sampleType,ask=True)
        self.getDirectory()
            
    def _basicStructure(self):
        """
        Returns the basic form of filter_function.py
        """
        return 'import numpy as np\n\ndef starFilter(data):\n\t"""\n\t{0}\n\t"""\n\treturn'.format(self.name)

    def _sampleInfo(self):
        """
        Retrieves information about the sample from the user.
        """
        while not self.done:
            key = raw_input('Data key: ')
            # Check if key is accepted
            if key in _keyList:
                self.name+='_'+key
                # Get info for this key
                match = self._match(key)
                if match[0]=='done':
                    self.done = True
                    break
                elif match[0]=='a':
                    self.name+='_fullsample'
                    self.condition = 'np.where(data)'
                    self.done=True
                elif match[0]=='m':
                    # Add string form of the matching condition and update the name
                    self.name+='_match'+match[1]
                    self.condition += ' (data[\'{0}\'] == "{1}") &'.format(key,match[1])
                elif match[0]=='s':
                    # Add string form of the slicing condition and update the name
                    self.name+='_up'+str(match[1])+'_lo'+str(match[2])
                    self.condition += ' (data[\'{0}\'] < {1}) & (data[\'{0}\'] > {2}) &'.format(key,match[1],match[2])
            # If key not accepted, make recursive call
            elif key not in _keyList and key != 'done':
                print 'Got a bad key. Try choosing one of ',_keyList
                self._sampleInfo()
                # If done condition, exit
            elif key == 'done':
                print 'Done getting filter information'
                self.done = True
                break
    

    def _match(self,key):
        """
        Returns user-generated conditions to match or slice in a given key.

        key:   label of the data set
        
        """
        # Check whether we will match to key or slice in its range
        match = raw_input('Default is full range. Match or slice? ').strip()
        
        if match == 'match' or match == 'm' or match == 'Match':
            m = raw_input('Match value: ')
            if m=='done':
                print 'Done getting filter information'
                return 'done',None
            # Check if match value has at least one star, if not call _match recursively
            elif m!='done' and m in self.data[key]:
                return 'm',m
            elif m not in self.data[key]:
                print 'No match for this key. Try choosing one of ',np.unique(self.data[key])
                self._match(key)

        elif match == 'slice' or match == 's' or match == 'Slice':
            # Get limits of slice
            upperLimit = raw_input('Upper limit (Enter for maximum): ')
            lowerLimit = raw_input('Lower limit (Enter for minimum): ')
            if upperLimit == 'done' or lowerLimit == 'done':
                print 'Done getting filter information'
                return 'done',None
            elif upperLimit != 'done' and lowerLimit != 'done':
                if upperLimit == 'max' or upperLimit == 'm' or upperLimit == '':
                    upperLimit = np.max(self.data[key])
                if lowerLimit == 'min' or lowerLimit == 'm' or lowerLimit == '':
                    lowerLimit = np.min(self.data[key])
                # Check limits are good - if not, call _match recursively
                try:
                    if float(upperLimit) <= float(lowerLimit):
                        print 'Limits are the same or are in the wrong order. Try again.'
                        self._match(key)
                    elif float(upperLimit) > float(lowerLimit):
                        print 'Found good limits'
                        return 's',float(upperLimit),float(lowerLimit)
                except ValueError as e:
                    print 'Please enter floats for the limits'
                    self._match(key)

        elif match == 'all' or match == 'a' or match == 'All':
            return 'a',None
            
            
        elif match == 'done':
            print 'Done getting filter information'
            return 'done',None

        else:
            print 'Please type m, s or a'
            self._match(key)
        
    def getDirectory(self):
        """
        Create directory for given filter
        """
        if not os.path.isdir(self.name):
            os.system('mkdir {0}'.format(self.name))
        return




class subStarSample(makeFilter):
    def __init__(self,sampleType,ask=True,correction=None):
        """
        Create a subsample according to a starFilter function
        Set ask=True if you wish to be asked about the filter, leave False to use existing filter

        """
        makeFilter.__init__(self,sampleType,ask=ask)
        import filter_function
        reload(filter_function)
        from filter_function import starFilter
        self._matchingStars = starFilter(self.data)
        self.matchingData = self.data[self._matchingStars]
        self.numberStars = len(self.matchingData)
        self.correctUncertainty()
        
        
    def checkArrays(self):
        if os.path.isfile(self.name+'/inputdata.pkl'):
            self.teff,self.logg,self.fe_h,self.spectra,self.spectra_errs,self._bitmasks = acs.pklread(self.name+'/inputdata.pkl')
        elif not os.path.isfile(self.name+'/inputdata.pkl'):
            self.makeArrays(self.matchingData)
            inputdata = [self.teff,self.logg,self.fe_h,self.spectra,self.spectra_errs,self._bitmasks]
            acs.pklwrite(self.name+'/inputdata.pkl',inputdata)

    def correctUncertainty(self,correction=None):
        self.checkArrays()
        SNR = self.spectra/self.spectra_errs
        if isinstance(correction,(str)):
            correction = acs.pklread(correction)
        if isinstance(correction,(float,int)):
            self.spectra_errs *= np.sqrt(correction)
        elif isinstance(correction,(list)):
            correction = np.array(correction)
        if isinstance(correction,(np.ndarray)):
            if correction.shape != self.spectra_errs.shape:
                correction = np.tile(correction,(self.spectra_errs.shape[0],1))
            self.spectra_errs = np.sqrt(correction*self.spectra_errs**2)
        

    def imshow(self,plotData,saveName=None,title = '',xlabel='pixels',ylabel='stars',**kwargs):
        plt.imshow(plotData,interpolation='nearest',aspect = float(plotData.shape[1])/plotData.shape[0],**kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        if saveName=='title':
            sname = title.split(' ')
            sname = '_'.join(sname)
            plt.savefig(self.name+'/'+sname+'.png')
            plt.close()
        elif saveName:
            plt.savefig(self.name+'/'+saveName+'.png')
            plt.close()
            


class mask(subStarSample):
    """
    Define a mask given a conditions in the form of maskConditions
    """
    def __init__(self,sampleType,maskFilter,ask=True,correction=None):
        subStarSample.__init__(self,sampleType,ask=ask)
        self._SNR = self.spectra/self.spectra_errs
        self.masked = np.zeros(self.spectra.shape).astype(bool)
        self.unmasked = np.ones(self.spectra.shape).astype(bool)
        self._maskHere = maskFilter(self)
        self.masked[self._maskHere]= True
        self.unmasked[self._maskHere] = False
        self.applyMask()

    def applyMask(self):
        """
        Mask all arrays according to maskConditions

        """
        self.teff.mask = self.masked
        self.logg.mask = self.masked
        self.fe_h.mask = self.masked

        self.spectra.mask = self.masked
        self.spectra_errs.mask = self.masked

        self.keywordMap = {'TEFF':self.teff,
                           'LOGG':self.logg,
                           'FE_H':self.fe_h
                       }


class fit(mask):
    def __init__(self,sampleType,maskFilter,ask=True,correction=None,degree=2):
        mask.__init__(self,sampleType,maskFilter,ask=ask)
        self.polynomial = PolynomialFeatures(degree=degree)
        self.findResiduals()

    def makeMatrix(self,pixel):
        numberStars = len(self.spectra[:,pixel][self.unmasked[:,pixel]])
        indeps = np.zeros((len(independentVariables[self._sampleType]),numberStars))
        for i in range(len(independentVariables[self._sampleType])):
            variable = independentVariables[self._sampleType][i]
            indeps[i] = self.keywordMap[variable][:,pixel][self.unmasked[:,pixel]]
        return np.matrix(self.polynomial.fit_transform(indeps.T))

    def findFit(self,pixel):
        indeps = self.makeMatrix(pixel)
        covInverse = np.diag(1./self.spectra_errs[:,pixel][self.unmasked[:,pixel]]**2)
        starsAtPixel = np.matrix(self.spectra[:,pixel][self.unmasked[:,pixel]])
        newIndeps = covInverse*indeps
        newIndeps = indeps.T*newIndeps
        newStarsAtPixel = covInverse*starsAtPixel.T
        newStarsAtPixel = indeps.T*newStarsAtPixel
        linModel = linear_model.LinearRegression()
        linModel.fit(newIndeps,newStarsAtPixel)
        return indeps*linModel.coef_.T,linModel.coef_

    def multiFit(self,minStarNum='default'):
        self.testM = self.makeMatrix(0)
        if minStarNum=='default':
            self.minStarNum = self.testM.shape[1]+1
        elif minStarNum!='default':
            self.minStarNum = minStarNum
        self.fitCoeffs = np.ma.masked_array(np.zeros((aspcappix,self.testM.shape[1])))
        self.fitSpectra = np.ma.masked_array(np.zeros((self.spectra.shape)),mask = self.spectra.mask)
        for pixel in range(aspcappix):
            if np.sum(self.unmasked[:,pixel].astype(int)) < self.minStarNum:
                self.fitSpectra[:,pixel].mask = np.ones(self.fitSpectra[:,pixel].shape)
                self.fitCoeffs[pixel].mask = np.ones(self.fitCoeffs[pixel].shape)
                self.unmasked[:,pixel] = np.zeros(self.unmasked[:,pixel].shape)
                self.masked[:,pixel] = np.ones(self.masked[:,pixel].shape)
            else:
                fitSpectrum,coefficients = self.findFit(pixel)
                self.fitSpectra[:,pixel][self.unmasked[:,pixel]] = fitSpectrum
                self.fitCoeffs[pixel] = coefficients
        self.applyMask()
    
    def findResiduals(self,minStarNum='default'):
        self.multiFit(minStarNum=minStarNum)
        self.residuals = self.spectra - self.fitSpectra 

class EMPCA(fit):
    def __init__(self,sampleType,maskFilter,ask=True,correction=None,degree=2,nvecs=5,deltR2=0,mad=False):
        fit.__init__(self,sampleType,maskFilter,ask=ask,correction=correction,degree=degree)
        self.nvecs=nvecs
        self.deltR2=deltR2
        self.mad=mad
        self.pixelEMPCA()
        self.setR2noise()

    def pixelEMPCA(self,randomSeed=1):
        self.goodPixels=([i for i in range(aspcappix) if np.sum(self.residuals[:,i].mask) < self.residuals.shape[0]-self.minStarNum],)
        empcaResiduals = self.residuals.T[self.goodPixels].T
        unmasked = (empcaResiduals.mask==False)
        basicWeights=unmasked.astype(float)
        self.empcaModel = empca(empcaResiduals.data,weights=basicWeights,nvec=self.nvecs,deltR2=self.deltR2,mad=self.mad,randseed=randomSeed)
        self.setR2(self.empcaModel)
        self.resizePixelEigvec(self.empcaModel)
        self.elementEigVec(self.empcaModel)
        errorWeights = basicWeights
        errorWeights[unmasked] = 1./(self.spectra_errs.T[self.goodPixels].T[unmasked])**2
        self.empcaModelWeight = empca(empcaResiduals.data,weights=errorWeights,nvec=self.nvecs,deltR2=self.deltR2,mad=self.mad,randseed=randomSeed)
        self.setR2(self.empcaModelWeight)
        self.resizePixelEigvec(self.empcaModelWeight)
        self.elementEigVec(self.empcaModel)

    def setR2(self,model):
        vecs = len(model.eigvec)
        R2Array = np.zeros(vecs+1)
        for vec in range(vecs+1):
            R2Array[vec] = model.R2(vec,mad=self.mad)
        model.R2Array = R2Array

    def resizePixelEigvec(self,model):
        neweigvec = np.ma.masked_array(np.zeros((self.nvecs,aspcappix)))
        for vec in range(self.nvecs):
            newvec = np.ma.masked_array(np.zeros((aspcappix)),mask=np.ones(aspcappix))
            newvec[self.goodPixels] = model.eigvec[vec][:len(self.goodPixels[0])]
            newvec.mask[self.goodPixels] = 0
            neweigvec[vec] = newvec/np.sqrt(np.sum(newvec**2))
        model.eigvec = neweigvec

    def elementEigVec(self,model):
        neweigvec = np.zeros((self.nvecs,len(elems)))
        for ind in range(len(elems)):
            window = elemwindows[elems[ind]]
            normWindow = np.ma.masked_array(window/np.sqrt(np.sum(window**2)))
            for vec in range(self.nvecs):
                neweigvec[vec][ind] = np.ma.sum(normWindow*model.eigvec[vec])
        for vec in range(self.nvecs):
            neweigvec[vec] = neweigvec[vec]/np.sqrt(np.sum(neweigvec[vec]**2))
        model.elementEigVec = neweigvec
                            
    def setR2noise(self):
        if self.mad:
            self.Vdata = self.empcaModelWeight._unmasked_data_mad2*1.4826**2
        elif not self.mad:
            self.Vdata = self.empcaModelWeight._unmasked_data_var
        self.Vnoise = np.mean(1./(self.empcaModelWeight.weights[self.empcaModelWeight.weights!=0]))
        self.R2noise = 1.-(self.Vnoise/self.Vdata)

    def plotEigVec(self,model,index):
        plt.ylim(-1,1)
        plt.axhline(0,color='k')
        plt.plot(model.elementEigVec[index],'o-',lw=3)
        plt.xlabel('Element')
        plt.ylabel('Eigenvector {0}'.format(index))
        plt.xticks(range(len(elems)),elems)
        
        
