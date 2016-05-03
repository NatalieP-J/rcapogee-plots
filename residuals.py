import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.nonparametric.smoothers_lowess as sm
import scipy as sp
from tqdm import tqdm
import data
reload(data)
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
    

class fakeEMPCA(object):
    """
    
    Class to contain crucial EMPCA-related objects.

    """
    def __init__(self,data,weights,eigenvectors,coeff,mad=False):
        self.data = data
        self.weights = weights
        self.mad = mad
        ii = np.where(self.weights > 0)
        self._unmasked = ii
        self._unmasked_data_var = np.var(self.data[ii])
        self._unmasked_data_mad2 = np.sum(np.median(np.fabs(self.data[ii]\
                                                      -np.median(self.data[ii])))**2.)
        self.eigvec = eigenvectors
        self.coeff = coeff

    def R2(self,nvec=None,mad=False):
        mx = np.zeros(self.data.shape)
        for i in range(nvec):
            mx += np.outer(self.coeff[:,i],self.eigvec[i])
        d = mx-data

        #- Only consider R2 for unmasked data
        if mad:
            med= np.median(d[self._unmasked])
            return 1.0 - \
                np.sum(np.median(np.fabs(d-med)[self._unmasked])**2.)/\
                (self._unmasked_data_mad2)        
        else:
            print np.var(d[self._unmasked]),self._unmasked_data_var
            return 1.0 - np.var(d[self._unmasked]) / self._unmasked_data_var


class smallEMPCA(object):
    def __init__(self,R2Array,R2noise,mad,numpix,eigvec,coeff):
        self.R2Array = R2Array
        self.R2noise = R2noise
        self.mad = mad
        self.numpix = numpix
        self.eigvec = eigvec
        self.coeff = coeff

def pixel2element(arr):
    if arr.shape[1] == aspcappix:
        return np.dot(arr,normwindows.T)
    elif arr.shape[0] == aspcappix:
        return np.dot(arr.T,normwindow.T)
    

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

# Example mask filter function
def maskFilter(sample):
    """
    Returns True where sample properties match conditions
    
    sample:   an object of mask class

    """
    sample.spectra_errs[sample._SNR>200] = sample.spectra[sample._SNR>200]/200.
    maskbits = bm.bits_set(badcombpixmask)
    return (sample._SNR < 50.) | bitsNotSet(sample._bitmasks,maskbits)

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
            frac = numpix/len(xarray)
            low_smooth = sm.lowess(yarray[array_mask],xarray[array_mask],
                                   frac=frac,it=3,return_sorted=False)
        smooth_interp = sp.interpolate.interp1d(xarray[array_mask],
                                                low_smooth,bounds_error=False)
        smoothmedian[detectors[i]:detectors[i+1]] = smooth_interp(xarray)
    nanlocs = np.where(np.isnan(smoothmedian))
    smoothmedian[nanlocs] = 1
    return smoothmedian


class starSample(object):
    """
    Gets properties of a sample of stars given a key that defines the 
    read function.
    
    """
    def __init__(self,sampleType):
        """
        Get properties for all stars that match the sample type

        sampleType:   designator of the sample type - must be a key in readfn 
                      and independentVariables in data.py
        
        """
        self._sampleType = sampleType
        self._getProperties()

    def _getProperties(self):
        """
        Get properties of all possible stars to be used.
        """
        self.data = readfn[self._sampleType]()

    def initArrays(self,data):
        """
        Initialize arrays.
        """
        # Create fit variable arrays
        self.teff = np.ma.masked_array(np.zeros((len(data)),
                                                dtype=float))
        self.logg = np.ma.masked_array(np.zeros((len(data)),
                                                dtype=float))
        self.fe_h = np.ma.masked_array(np.zeros((len(data)),
                                                dtype=float))
        
        # Create spectra arrays
        self.spectra = np.ma.masked_array(np.zeros((len(data),aspcappix),
                                                   dtype=float))
        self.spectra_errs = np.ma.masked_array(np.zeros((len(data),aspcappix),
                                                        dtype=float))
        self._bitmasks = np.zeros((len(data),aspcappix),dtype=np.int64)
        
    def makeArrays(self,data):
        """
        Create arrays across all stars in the sample with shape number of 
        stars by aspcappix.
        
        data:   array whose columns contain information about stars in sample
        
        """
        
        self.initArrays(data)
        
        # Fill arrays for each star
        for star in tqdm(range(len(data)),desc='read star data'):
            LOC = data[star]['LOCATION_ID']
            APO = data[star]['APOGEE_ID']
            TEFF = data[star]['TEFF']
            LOGG = data[star]['LOGG']
            FE_H = data[star]['FE_H']
            
            # Fit variables
            self.teff[star] = np.ma.masked_array(TEFF)
            self.logg[star] = np.ma.masked_array(LOGG)
            self.fe_h[star] = np.ma.masked_array(FE_H)
            
            # Spectral data
            self.spectra[star] = apread.aspcapStar(LOC,APO,ext=1,header=False, 
                                                   aspcapWavegrid=True)
            self.spectra_errs[star] = apread.aspcapStar(LOC,APO,ext=2,
                                                        header=False, 
                                                        aspcapWavegrid=True)
            self._bitmasks[star] = apread.apStar(LOC,APO,ext=3, header=False, 
                                                 aspcapWavegrid=True)[1] 
            
    def plotHistogram(self,array,title = '',xlabel = '',
                      ylabel = 'number of stars',saveName=None,**kwargs):
        """
        Plots a histogram of some input array, with the option to save it.
        
        array:      array to plot as histogram
        title:      (optional) title of the plot
        xlabel:     (optional) x-axis label of the plot
        ylabel:     y-axis label of the plot (default: 'number of stars')
        saveName:   (optional) path to save plot, without file extension
        **kwargs:   kwargs for numpy.histogram
        
        """
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
    Contains functions to create a filter and associated directory 
    name for a starSample.
    """
    def __init__(self,sampleType,ask=True):
        """
        Sets up filter_function.py file to contain the appropriate function 
        and puts the save directory name in the docstring of the function.

        sampleType:   designator of the sample type - must be a key in readfn 
                      and independentVariables in data.py
        ask:          if True, function asks for user input to make 
                      filter_function.py, if False, uses existing 
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
            # Check that the user set conditions. 
            # If conditions not set, recursively call init
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
            # Import existing filter function. If function doesn't exist, 
            # recursively call init
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
                    # Add string form of the matching condition and 
                    # update the name
                    self.name+='_match'+match[1]
                    self.condition += ' (data[\'{0}\'] == "{1}") &'.format(key,match[1])
                elif match[0]=='s':
                    # Add string form of the slicing condition and 
                    # update the name
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

        key:   label of property of the data set
        
        """
        # Check whether we will match to key or slice in its range
        match = raw_input('Default is full range. Match or slice? ').strip()
        
        if match == 'match' or match == 'm' or match == 'Match':
            m = raw_input('Match value: ')
            if m=='done':
                print 'Done getting filter information'
                return 'done',None
            # Check if match value has at least one star, 
            # if not call _match recursively
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

        # Option to use the entire sample
        elif match == 'all' or match == 'a' or match == 'All':
            return 'a',None
            
        # Exit filter finding 
        elif match == 'done':
            print 'Done getting filter information'
            return 'done',None

        # Invalid entry condition
        else:
            print 'Please type m, s or a'
            self._match(key)
        
    def getDirectory(self):
        """
        Create directory to store results for given filter.
        
        """
        if not os.path.isdir(self.name):
            os.system('mkdir {0}'.format(self.name))
        return


class subStarSample(makeFilter):
    """
    Given a filter function, defines a subsample of the total sample of stars.
    
    """
    def __init__(self,sampleType,ask=True):
        """
        Create a subsample according to a starFilter function
        
        sampleType:   designator of the sample type - must be a key in readfn 
                      and independentVariables in data.py
        ask:          if True, function asks for user input to make 
                      filter_function.py, if False, uses existing 
                      filter_function.py
        
        """
        # Create starFilter
        makeFilter.__init__(self,sampleType,ask=ask)
        import filter_function
        reload(filter_function)
        from filter_function import starFilter
        # Find stars that satisfy starFilter and cut data accordingly
        self._matchingStars = starFilter(self.data)
        self.matchingData = self.data[self._matchingStars]
        self.numberStars = len(self.matchingData)
        self.checkArrays()
        
    def checkArrays(self):
        """
        Check if input data has already been saved as arrays. 
        If not, create them.
        
        """
        
        fnames = np.array([self.name+'/teff.npy',
                           self.name+'/logg.npy',
                           self.name+'/fe_h.npy',
                           self.name+'/spectra.npy',
                           self.name+'/spectra_errs.npy',
                           self.name+'/bitmasks.npy'])
        fexist = True
        for f in fnames:
            fexist *= os.path.isfile(f)
        if fexist:
            self.initArrays(data)
            self.teff.data = np.load(self.name+'/teff.npy')
            self.logg.data = np.load(self.name+'/logg.npy')
            self.fe_h.data = np.load(self.name+'fe_h.npy')
            self.spectra.data = np.load(self.name+'/spectra.npy')
            self.spectra_errs.data = np.load(self.name+'/spectra_errs.npy')
            self._bitmasks.data = np.load(self.name+'/bitmasks.npy')
            
        elif not fexist:
            self.makeArrays(self.matchingData)
            np.save(self.name+'/teff.npy',self.teff.data)
            np.save(self.name+'/logg.npy',self.logg.data)
            np.save(self.name+'fe_h.npy',self.fe_h.data)
            np.save(self.name+'/spectra.npy',self.spectra.data)
            np.save(self.name+'/spectra_errs.npy',self.spectra_errs.data)
            np.save(self.name+'/bitmasks.npy',self._bitmasks.data)
            

    def correctUncertainty(self,correction=None):
        """
        Performs a correction on measurement uncertainty.

        correction:   Information on how to perform the correction.
                      May be a path to a pickled file, a float, or 
                      list of values.

        """
        self.checkArrays()
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

    def uncorrectUncertainty(self,correction=None):
        """
        Undoes correction on measurement uncertainty.

        correction:   Information on how to perform the correction.
                      May be a path to a pickled file, a float, or 
                      list of values.
        """
        self.checkArrays()
        if isinstance(correction,(str)):
            correction = acs.pklread(correction)
        if isinstance(correction,(float,int)):
            self.spectra_errs /= np.sqrt(correction)
        elif isinstance(correction,(list)):
            correction = np.array(correction)
        if isinstance(correction,(np.ndarray)):
            if correction.shape != self.spectra_errs.shape:
                correction = np.tile(correction,(self.spectra_errs.shape[0],1))
            self.spectra_errs = np.sqrt(self.spectra_errs**2/correction)

    def imshow(self,plotData,saveName=None,title = '',xlabel='pixels',ylabel='stars',**kwargs):
        """
        Creates a square 2D plot of some input array, with the 
        option to save it.

        plotData:   2D array to plot
        saveName:   (optional) path to save plot without file extension
        title:      (optional) title for the plot
        xlabel:     x-axis label for the plot (default:'pixels')
        ylabel:     y-axis label for the plot (default:'stars')
        **kwargs:   kwargs for matplotlib.pyplot.imshow
        
        """
        plt.imshow(plotData,interpolation='nearest',
                   aspect = float(plotData.shape[1])/plotData.shape[0],**kwargs)
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

    def logplot(self,arrays,labels,ylabel='log10 of coefficients',xlabel='coefficient',reshape=True,
                coeff_labels=True):
        """
        Creates a plot with log values of input array, with legend indicating where array is positive
        and where its negative.

        arrays:         List of input arrays
        labels:         Legend labels for the input arrays
        ylabel:         Label for y-axis
        xlabel:         Label for x-axis
        reshape:        If True, reshape input arrays (use if one of the inputs is a numpy matrix)
        coeff_labels:   If True, use the default list of labels for the x-axis fit coefficients
        """
        if not isinstance(arrays,(list,np.ndarray)):
            arrays = [arrays]
        plt.figure(figsize=(14,8))
        for a in range(len(arrays)):
            array = arrays[a]
            if reshape:
                array = np.reshape(np.array(arrays[a]),(len(coeff_inds[self._sampleType]),))
            # Find independent indices
            x = np.arange(len(array))
            # Find where positive and negative
            pos = np.where(array>0)
            neg = np.where(array<0)
            # Create legend labels
            poslabel = 'positive {0}'.format(labels[a])
            neglabel = 'negative {0}'.format(labels[a])
            # Plot positive and negative separately
            plt.plot(x[pos],np.log10(array[pos]),'o',label=poslabel)
            plt.plot(x[neg],np.log10(np.fabs(array[neg])),'o',label=neglabel)
        # Extend limits so all points are clearly visible
        plt.xlim(x[0]-1,x[-1]+1)
        # Label axes
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        # Label x-ticks if requested
        if coeff_labels:
            plt.xticks(range(len(coeff_inds[self._sampleType])),coeff_inds[self._sampleType])
        # Draw legend
        plt.legend(loc='best')
        plt.show()
            


class mask(subStarSample):
    """
    Define and apply a mask given a set of conditions in the form of the
    maskConditions function.
    
    """
    def __init__(self,sampleType,maskFilter,ask=True):
        """
        Mask a subsample according to a maskFilter function
        
        sampleType:   designator of the sample type - must be a key in readfn 
                      and independentVariables in data.py
        maskFilter:   function that decides on elements to be masked
        ask:          if True, function asks for user input to make 
                      filter_function.py, if False, uses existing 
                      filter_function.py
        
        """
        subStarSample.__init__(self,sampleType,ask=ask)
        self._SNR = self.spectra/self.spectra_errs
        # create default mask arrays (mask nothing)
        self.masked = np.zeros(self.spectra.shape).astype(bool)
        self.unmasked = np.ones(self.spectra.shape).astype(bool)
        # find indices that should be masked
        self._maskHere = maskFilter(self)
        # update mask arrays
        self.masked[self._maskHere]= True
        self.unmasked[self._maskHere] = False
        # apply mask arrays to data
        self.applyMask()

    def applyMask(self):
        """
        Mask all arrays according to maskConditions

        """
        # spectral information
        self.spectra.mask = self.masked
        self.spectra_errs.mask = self.masked

        # create dictionary tracing the independent variables to keywords
        self.keywordMap = {'TEFF':self.teff,
                           'LOGG':self.logg,
                           'FE_H':self.fe_h
                       }


class fit(mask):
    """
    Contains functions to find polynomial fits.
    
    """
    def __init__(self,sampleType,maskFilter,ask=True,degree=2):
        """
        Fit a masked subsample.
        
        sampleType:   designator of the sample type - must be a key in readfn 
                      and independentVariables in data.py
        maskFilter:   function that decides on elements to be masked
        ask:          if True, function asks for user input to make 
                      filter_function.py, if False, uses existing 
                      filter_function.py
        degree:       degree of polynomial to fit
        
        """
        mask.__init__(self,sampleType,maskFilter,ask=ask)
        # create a polynomial object to be used in producing independent 
        # variable matrix
        self.degree = degree
        self.noncrossInds = ([0,1,2,3,4,7,9])
        self.crossInds = ([5,6,8],)
        self.polynomial = PolynomialFeatures(degree=degree)
        self.testM = self.makeMatrix(0)

    def makeMatrix(self,pixel):
        """
        Find independent variable matrix
        
        pixel:   pixel to use, informs the mask on the matrix
        
        Returns the matrix
        """
        # Find the number of unmasked stars at this pixel
        numberStars = len(self.spectra[:,pixel][self.unmasked[:,pixel]])
        
        # Create basic independent variable array
        indeps = np.zeros((numberStars,
                           len(independentVariables[self._sampleType])))

        for i in range(len(independentVariables[self._sampleType])):
            variable = independentVariables[self._sampleType][i]
            indep = self.keywordMap[variable][self.unmasked[:,pixel]]
            indeps[:,i] = indep-np.median(indep)
        # use polynomial to produce matrix with all necessary columns
        return np.matrix(self.polynomial.fit_transform(indeps))

    def findFit(self,pixel,eigcheck=False):
        """
        Fits polynomial to all spectra at a given pixel, weighted by spectra 
        uncertainties.

        pixel:   pixel at which to perform fit

        Return polynomial values and coefficients

        """
        # find matrix for polynomial of independent values
        indeps = self.makeMatrix(pixel)
        self.numparams = indeps.shape[1]
        # calculate inverse covariance matrix
        covInverse = np.diag(1./self.spectra_errs[:,pixel][self.unmasked[:,pixel]]**2)
        # find matrix for spectra values
        starsAtPixel = np.matrix(self.spectra[:,pixel][self.unmasked[:,pixel]])
        
        # transform to matrices that have been weighted by the inverse 
        # covariance
        newIndeps = np.dot(indeps.T,np.dot(covInverse,indeps))

        # Degeneracy check
        degen = False
        if eigcheck:
            eigvals,eigvecs = np.linalg.eig(newIndeps)
            if any(np.fabs(eigvals)<1e-10) and indeps.shape[1] > self.degree+1:
                print 'degenerate pixel ',pixel,' coeffs ',np.where(np.fabs(eigvals) < 1e-10)[0] 
                degen = True
                indeps = indeps.T[self.noncrossInds].T
        
        newStarsAtPixel = np.dot(indeps.T,np.dot(covInverse,starsAtPixel.T))
        invNewIndeps = np.linalg.inv(newIndeps)

        # calculate fit coefficients
        coeffs = np.dot(invNewIndeps,newStarsAtPixel)
        #coeffs = np.linalg.lstsq(newIndeps,newStarsAtPixel)[0]
        coeff_errs = np.array([np.sqrt(np.array(invNewIndeps)[i][i]) for i in range(newIndeps.shape[1])])
        bestFit = indeps*coeffs
        # If degeneracy, make coefficients into the correct shape
        if degen:
            newcoeffs = np.ma.masked_array(np.zeros(self.numparams),
                                           mask = np.zeros(self.numparams))
            newcoeff_errs = np.ma.masked_array(np.zeros(self.numparams),
                                               mask = np.zeros(self.numparams))
            newcoeffs[self.noncrossInds] = coeffs
            newcoeff_errs[self.noncrossInds] = coeff_errs
            newcoeffs.mask[self.crossInds] = True
            newcoeff_errs.mask[self.crossInds] = True
            coeffs = newcoeffs
            coeff_errs = newcoeff_errs
        return bestFit,coeffs.T,coeff_errs

    def multiFit(self,minStarNum='default',eigcheck=False):
        """
        Loop over all pixels and find fit. Mask where there aren't enough 
        stars to fit.

        minStarNum:   (optional) number of stars required to perform fit 
                      (default:'default' which sets minStarNum to the number 
                       of fit parameters plus one)
        
        """
        # create sample matrix to confirm the number of parameters
        self.testM = self.makeMatrix(0)
        self.numparams = self.testM.shape[1]
        
        # set minimum number of stars needed for the fit
        if minStarNum=='default':
            self.minStarNum = self.testM.shape[1]+1
        elif minStarNum!='default':
            self.minStarNum = minStarNum
        
        # create arrays to hold fit results
        self.fitCoeffs = np.ma.masked_array(np.zeros((aspcappix,
                                                      self.numparams)))
        self.fitCoeffErrs = np.ma.masked_array(np.zeros((aspcappix,
                                                         self.numparams)))
        self.fitSpectra = np.ma.masked_array(np.zeros((self.spectra.shape)),
                                             mask = self.spectra.mask)
        
        # perform fit at all pixels with enough stars
        for pixel in tqdm(range(aspcappix),desc='fit'):
            if np.sum(self.unmasked[:,pixel].astype(int)) < self.minStarNum:
                # if too many stars missing, update mask
                self.fitSpectra[:,pixel].mask = np.ones(self.spectra.shape[1])
                self.fitCoeffs[pixel].mask = np.ones(self.numparams)
                self.fitCoeffErrs[pixel].mask = np.ones(self.numparams)
                self.unmasked[:,pixel] = np.zeros(self.unmasked[:,pixel].shape)
                self.masked[:,pixel] = np.ones(self.masked[:,pixel].shape)
            else:
                # if fit possible update arrays
                fitSpectrum,coefficients,coefficient_uncertainty = self.findFit(pixel,eigcheck)
                self.fitSpectra[:,pixel][self.unmasked[:,pixel]] = fitSpectrum
                self.fitCoeffs[pixel] = coefficients
                self.fitCoeffErrs[pixel] = coefficient_uncertainty

        # update mask on input data
        self.applyMask()

    def fitStatistic(self):
        """
        Adds to fit object chi squared and reduced chi squared properties.

        """
        self.fitChiSquared = np.ma.sum((self.spectra-self.fitSpectra)**2/self.spectra_errs**2,axis=0)
        # Calculate degrees of freedom
        if isinstance(self.fitCoeffs.mask,np.ndarray):
            dof = self.numberStars - np.sum(self.fitCoeffs.mask==False,axis=1) - 1
        else:
            dof = self.numberStars - self.numparams - 1
        self.fitReducedChi = self.fitChiSquared/dof
    
    def findResiduals(self,minStarNum='default',gen=True):
        """
        Calculate residuals from polynomial fits.
        
        minStarNum:   (optional) number of stars required to perform fit 
                      (default:'default' which sets minStarNum to the number 
                       of fit parameters plus one)
        
        """
        if gen:
            self.multiFit(minStarNum=minStarNum)
            self.residuals = self.spectra - self.fitSpectra 
            acs.pklwrite(self.name+'/residuals.pkl',self.residuals)
        if not gen:
            self.testM = self.makeMatrix(0)
            self.minStarNum = self.testM.shape[1]+1
            self.residuals = acs.pklread(self.name+'/residuals.pkl')

    def findAbundances(self):
        """
        From calculated residuals, calculate elemental abundances for each star.
        """
        self.abundances = pixel2element(self.residuals)
        self.abundance_errs = np.sqrt(np.dot(self.spectra_errs**2,normwindows.T**2))


    def plotAbundances2d(self,elem1,elem2,saveName=None):
        """
        Creates a 2D plot comparing two abundances.

        elem1:      Name of first element (eg. 'C')
        elem2:      Name of second element
        saveName:   Name of output file (saved in directory associated with sample)
        """
        ind1 = elems.index(elem1)
        ind2 = elems.index(elem2)
        plt.figure(figsize=(10,8))
        plt.plot(self.abundances[:,ind1],self.abundances[:,ind2],'o')
        plt.xlabel(elem1)
        plt.ylabel(elem2)
        if saveName:
            plt.savefig(self.name+'/'+saveName+'.png')

    def plotAbundances3d(self,elem1,elem2,elem3,saveName=None):
        """
        Creates a 3D plot comparing two abundances.

        elem1:      Name of first element (eg. 'C')
        elem2:      Name of second element
        elem3:      Name of third element
        saveName:   Name of output file (saved in directory associated with sample)
        """
        ind1 = elems.index(elem1)
        ind2 = elems.index(elem2)
        ind3 = elems.index(elem3)
        # turn on interactivity to rotate the plot
        plt.ion()
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111,projection='3d')  
        ax.scatter(self.abundances[:,ind1],self.abundances[:,ind2],self.abundances[:,ind3])
        ax.set_xlabel(elem1)
        ax.set_ylabel(elem2)
        ax.set_zlabel(elem3)
        if saveName:
            plt.savefig(self.name+'/'+saveName+'.png')


    def testFit(self,errs=None,randomize=False, params=defaultparams, singlepix=None,minStarNum='default'):

        """
        Test fit accuracy by generating a data set from given input parameters.

        errs:        Sets errors to use in the fit. If a int or float, set 
                     constant errors. Else, use data set errors.
        randomize:   Sets whether generated data gets Gaussian noise added,
                     drawn from flux uncertainty.
        params:      Either an array of parameters to use or an index to select
                     from existing fitCoeffs array.
        singlepix:   Sets whether to fit all pixels, or just a single one
                     (random if singlepix not set as integer)

        """
        if minStarNum=='default':
            self.minStarNum = self.testM.shape[1]+1
        elif minStarNum!='default':
            self.minStarNum = minStarNum
        
        # Choose parameters if necessary
        if isinstance(params,(int)):
            params = self.fitCoeffs[params]
        
        if isinstance(params,(dict)):
            params = params[self._sampleType]

        if isinstance(params,(list)):
            params = np.array(params)

        # Store parameters for future comparison
        self.testParams = np.matrix(params).T
        # Keep old spectra information
        self.old_spectra = np.ma.masked_array(np.zeros(self.spectra.shape))
        self.old_spectra[:] = self.spectra
        self.old_spectra_errs = np.ma.masked_array(np.zeros(self.spectra_errs.shape))
        self.old_spectra_errs[:] = self.spectra_errs

        if not errs:
            # assign errors as constant with consistent max
            self.spectra_errs = np.ma.masked_array(np.ones(self.old_spectra_errs.shape))
            self.spectra_errs.mask=self.old_spectra_errs.mask
            
        # Generate data 
        for pixel in range(aspcappix):
            if np.sum(self.unmasked[:,pixel].astype(int)) > self.minStarNum:
                indeps = self.makeMatrix(pixel)
                self.spectra[:,pixel][self.unmasked[:,pixel]] = np.reshape(np.array(indeps*self.testParams),self.spectra[:,pixel][self.unmasked[:,pixel]].shape)
         
        # If requested, added noise to the data        
        if randomize:
            self.spectra += self.spectra_errs*np.random.randn(self.spectra.shape[0],
                                                              self.spectra.shape[1])

        self.testParams = np.reshape(np.array(self.testParams),(len(params),))
        if not singlepix:
            # Calculate residuals
            self.multiFit()
            # Find the difference between the calculated parameters and original input
            self.diff = np.ma.masked_array([self.testParams-self.fitCoeffs[i] for i in range(len(self.fitCoeffs))],mask = self.fitCoeffs.mask)
        elif singlepix:
            fitSpectrum,coefficients,coefficient_uncertainty = self.findFit(singlepix)
            self.fitCoeffs = coefficients
            self.fitCoeffErrs = coefficient_uncertainty
            self.diff = self.testParams-coefficients
    
        # Normalize the difference by standard error size
        self.errNormDiff = self.diff/np.ma.median(self.spectra_errs)

        # Restore previous values
        self.spectra[:] = self.old_spectra
        self.spectra_errs[:] = self.old_spectra_errs


    def findCorrection(self,cov,median=True,numpix=10.,frac=None):
        """
        Calculates the diagonal of a square matrix and smooths it
        either over a fraction of the data or a number of elements,
        where number of elements takes precedence if both are set.

        cov:      Square matrix
        median:   If true, returns smoothed median, not raw diagonal
        numpix:   Number of elements to smooth over
        frac:     Fraction of data to smooth over

        Returns the diagonal
        """
        diagonal = np.ma.masked_array([cov[i,i] for i in range(len(cov))],
                                      mask=[cov.mask[i,i] for i in 
                                            range(len(cov))])
        if median:
            median = smoothMedian(diagonal,frac=frac,numpix=float(numpix))
            return median
        elif not median:
            return diagonal

    def pixelEMPCA(self,randomSeed=1,nvecs=5,deltR2=0,mad=False,correction=None,savename=None,gen=True,numpix=None,weighttype='basic'):
        """
        Calculates EMPCA on residuals in pixel space.

        randomSeed:   seed to initialize starting EMPCA vectors

        """
        if savename:
            try:
                self.empcaModelWeight = acs.pklread(self.name+'/'+savename)
            except IOError:
                gen = True
        if gen:
            self.numpix=numpix
            self.correctUncertainty(correction=correction)
            self.mad = mad
            self.nvecs = nvecs
            self.deltR2 = deltR2
            # Find pixels with enough stars to do EMPCA
            self.goodPixels=([i for i in range(aspcappix) if np.sum(self.residuals[:,i].mask) < self.residuals.shape[0]-self.minStarNum],)
            empcaResiduals = self.residuals.T[self.goodPixels].T
            
            # Calculate weights that just mask missing elements
            unmasked = (empcaResiduals.mask==False)
            basicWeights=unmasked.astype(float)

            # Find EMPCA model
            #self.empcaModel = empca(empcaResiduals.data,weights=basicWeights,
            #                        nvec=self.nvecs,deltR2=self.deltR2,mad=self.mad,
            #                        randseed=randomSeed)
        
            # Find R2 and R2noise for this model, and resize eigenvectors appropriately
            #self.setR2(self.empcaModel)
            #self.setR2noise(self.empcaModel) # This doesn't make sense since there are no errors
            #self.resizePixelEigvec(self.empcaModel)
            # Find eigenvectors in element space
            #eigvecs = self.elementEigVec(self.empcaModel)

            # Calculate weights from the inverse square of the flux uncertainty, keeping
            # zeros where residuals are masked
            errorWeights = np.zeros(basicWeights.shape)
            errorWeights[:] = basicWeights
            errorWeights[unmasked] = 1./((self.spectra_errs.T[self.goodPixels].T[unmasked])**2)
            self.empcaModelWeight = empca(empcaResiduals.data,weights=errorWeights,
                                          nvec=self.nvecs,deltR2=self.deltR2,
                                          mad=self.mad,randseed=randomSeed)    

            # Find R2 and R2noise for this model, and resize eigenvectors appropriately
            self.setR2(self.empcaModelWeight)
            self.setDeltaR2(self.empcaModelWeight)
            self.setR2noise(self.empcaModelWeight)
            self.resizePixelEigvec(self.empcaModelWeight)

            # Resize data set

            #newdata = np.zeros(self.residuals.T.shape)
            #newweights = np.zeros(self.residuals.T.shape)
            #newdata[self.goodPixels] = self.empcaModel.data.T
            #newweights[self.goodPixels] = self.empcaModel.weights.T
            #self.empcaModelWeight.data = newdata.T 
            #self.empcaModelWeight.weights = newweights.T


            # Find eigenvectors in element space
            #self.elementEigVec(self.empcaModelWeight)
            self.uncorrectUncertainty(correction=correction)
            self.smallModel = smallEMPCA(self.empcaModelWeight.R2Array,self.empcaModelWeight.R2noise,self.mad,self.numpix,self.empcaModelWeight.eigvec,self.empcaModelWeight.coeff)
            if savename:
                acs.pklwrite(self.name+'/'+savename,self.smallModel)

    def setR2(self,model):
        """
        Add R2 values for each eigenvector as array to model.

        model:   EMPCA model

        """
        vecs = len(model.eigvec)
        # Create R2 array
        R2Array = np.zeros(vecs+1)
        for vec in range(vecs+1):
            R2Array[vec] = model.R2(vec,mad=self.mad)
        # Add R2 array to model
        model.R2Array = R2Array

    def setDeltaR2(self,model):
        """
        """
        return (np.roll(model.R2Array,-1)-model.R2Array)[:-1]

    def resizePixelEigvec(self,model):
        """
        Resize eigenvectors to span full pixel space, masking where necessary.

        model:   EMPCA model

        """
        # Create array for reshape eigenvectors
        neweigvec = np.ma.masked_array(np.zeros((self.nvecs,aspcappix)))
        # Add eigenvectors to array with appropriate mask
        for vec in range(self.nvecs):
            newvec = np.ma.masked_array(np.zeros((aspcappix)),
                                        mask=np.ones(aspcappix))
            newvec[self.goodPixels] = model.eigvec[vec][:len(self.goodPixels[0])]
            newvec.mask[self.goodPixels] = 0
            # Normalize the vector
            neweigvec[vec] = newvec/np.sqrt(np.sum(newvec**2))
        # Change model eigenvectors to reshaped ones
        model.eigvec = neweigvec

    def elementEigVec(self,model):
        """
        Reduce eigenvector dimension to element space.

        model:   EMPCA model

        """
        neweigvec = pixel2element(model.eigvec)
        neweigvec /= np.tile(np.sum(neweigvec**2),(len(elems),1)).T
        return neweigvec
                            
    def setR2noise(self,model):
        """
        Calculate R2 noise, the threshold at which additional vectors are only
        explaining noise.

        model:   EMPCA model

        """
        # Determine which variance to use
        if self.mad:
            model.Vdata = model._unmasked_data_mad2*1.4826**2
        elif not self.mad:
            model.Vdata = model._unmasked_data_var
        # Calculate data noise
        model.Vnoise = np.mean(1./(model.weights[model.weights!=0]))
        # Calculate R2noise
        model.R2noise = 1.-(model.Vnoise/model.Vdata)

    def plotEigVec(self,model,index):
        """
        Plot an element space eigenvector from an EMPCA model.

        model:   EMPCA model
        index:   Index of eigenvector to plot

        """
        plt.ylim(-1,1)
        plt.axhline(0,color='k')
        plt.plot(model.eigvec[index],'o-',lw=3)
        plt.xlabel('Element')
        plt.ylabel('Eigenvector {0}'.format(index))
        plt.xticks(range(len(elems)),elems)
    
    def plotR2(self,model):
        """
        Plot R2 vs number of eigenvectors for an EMPCA model

        model:   EMPCA model
        
        """
        plt.ylim(0,1)
        plt.axhline(model.R2noise,color='k')
        plt.plot(model.R2Array)
        plt.ylabel('R2')
        plt.xlabel('number of eigvectors')

    def elementModel(self,model,mad=False):
        #self.resizePixelEigvec(model)
        eigvec = self.elementEigVec(model)
        data = pixel2element(model.data)
        weights = pixel2element(model.weights)
        coeff = np.zeros(model.coeff.shape)
        coeff[:] = model.coeff
        self.empcaElement = fakeEMPCA(data,weights,eigvec,coeff,mad=mad)
        self.setR2noise(self.empcaElement)
        self.setR2(self.empcaElement)

