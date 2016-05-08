import numpy as np
import os
from tqdm import tqdm
import data
reload(data)
from data import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    def initArrays(self,stardata):
        """
        Initialize arrays.
        """
        # Create fit variable arrays
        self.teff = np.ma.masked_array(np.zeros((len(stardata)),
                                                dtype=float))
        self.logg = np.ma.masked_array(np.zeros((len(stardata)),
                                                dtype=float))
        self.fe_h = np.ma.masked_array(np.zeros((len(stardata)),
                                                dtype=float))
        
        # Create spectra arrays
        self.spectra = np.ma.masked_array(np.zeros((len(stardata),aspcappix),
                                                   dtype=float))
        self.spectra_errs = np.ma.masked_array(np.zeros((len(stardata),
                                                         aspcappix),
                                                        dtype=float))
        self._bitmasks = np.zeros((len(stardata),aspcappix),dtype=np.int64)
        
    def makeArrays(self,stardata):
        """
        Create arrays across all stars in the sample with shape number of 
        stars by aspcappix.
        
        stardata:   array whose columns contain information about stars in 
                    sample
        
        """
        
        self.initArrays(stardata)
        
        # Fill arrays for each star
        for star in tqdm(range(len(stardata)),desc='read star data'):
            LOC = stardata[star]['LOCATION_ID']
            APO = stardata[star]['APOGEE_ID']
            TEFF = stardata[star]['TEFF']
            LOGG = stardata[star]['LOGG']
            FE_H = stardata[star]['FE_H']
            
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
            if key in keyList:
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
            elif key not in keyList and key != 'done':
                print 'Got a bad key. Try choosing one of ',keyList
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

    def directoryClean(self):
        """
        Removes all files from a specified directory.

        """
        os.system('rm -rf {0}/*'.format(self.name))


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
        # If all files exist, read data from file for increased initialization speed
        if fexist:
            self.teff = np.load(self.name+'/teff.npy')
            self.logg = np.load(self.name+'/logg.npy')
            self.fe_h = np.load(self.name+'/fe_h.npy')
            self.spectra = np.load(self.name+'/spectra.npy')
            self.spectra_errs = np.load(self.name+'/spectra_errs.npy')
            self._bitmasks = np.load(self.name+'/bitmasks.npy')
        # If any file is missing, generate arrays and write to file
        elif not fexist:
            self.makeArrays(self.matchingData)
            np.save(self.name+'/teff.npy',self.teff.data)
            np.save(self.name+'/logg.npy',self.logg.data)
            np.save(self.name+'/fe_h.npy',self.fe_h.data)
            np.save(self.name+'/spectra.npy',self.spectra.data)
            np.save(self.name+'/spectra_errs.npy',self.spectra_errs.data)
            np.save(self.name+'/bitmasks.npy',self._bitmasks)
            

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
            