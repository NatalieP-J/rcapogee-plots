import numpy as np
import os,inspect
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spectralspace.sample.access_spectrum as acs
import apogee.samples.rc as rcmodel
import apogee.tools.read as apread
from apogee.tools.path import change_dr
from spectralspace.sample.read_clusterdata import read_caldata
from importlib import reload
import isodist

font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  20
}

matplotlib.rc('font',**font)
plt.ion()

aspcappix = 7214

def rgsample():
    """
    Selects red giants from APOGEE sample
    """
    data= apread.allStar(main=True,exclude_star_bad=True,exclude_star_warn=True)
    jk= data['J0']-data['K0']
    z= isodist.FEH2Z(data['METALS'],zsolar=0.017)
    z[z > 0.024]= 0.024
    logg= data['LOGG']
    indx= ((jk >= 0.8)
            +(logg > rcmodel.loggteffcut(data['TEFF'],z,upper=True)))
    rgindx=indx*(data['METALS'] > -.8)
    return data[rgindx]

def get_synthetic(model,datadict=None,data=[],spectra=[],spectra_errs=[],bitmask=[]):
    """
    Retrieves information about a synthetic sample

    model:        star_sample object to fill with synthetic information
    datadict:     condensed argument containing entries for each of following kwargs. if the following
                  are passed separately they are overridden by the dictionary entries
    data:         numpy structured array with columns for TEFF, LOGG and FE_H, one entry per star
    spectra:      array of ASPCAP shaped spectra (7214 pixels per spectrum)
    spectra_errs: array of ASPCAP shaped uncertaintes on spectra (7214 pixels per spectrum)
    bitmask:      base 2 bitmask indicating flagged pixels. defaults to no flags

    Updates properties of model, returns nothing.

    """

    if isinstance(datadict,dict):
        data = datadict['data']
        spectra = datadict['spectra']
        spectra_errs = datadict['spectra_errs']
        bitmask = datadict['bitmask']
    model.data = data
    # Create fit variable arrays
    model.teff = np.ma.masked_array(data['TEFF'])
    model.logg = np.ma.masked_array(data['LOGG'])
    model.fe_h = np.ma.masked_array(data['FE_H'])
    model.c_h = np.ma.masked_array(data['C_H'])
    model.n_h = np.ma.masked_array(data['N_H'])
    model.o_h = np.ma.masked_array(data['O_H'])
    model.fib = np.ma.masked_array(data['MEANFIB'])

    # Create spectra arrays
    model.spectra = np.ma.masked_array(spectra)
    model.spectra_errs = np.ma.masked_array(np.zeros((len(data),
                                                    aspcappix)))
    if isinstance(spectra_errs,(int,float)):
        model.spectra_errs+=spectra_errs
    elif isinstance(spectra_errs,(np.ndarray)):
        model.spectra_errs += spectra_errs
    model._bitmasks = np.zeros((len(data),aspcappix),dtype=np.int64)
    if isinstance(bitmask,(np.ndarray)):
        model._bitmasks = bitmask


# Functions to access particular sample types
readfn = {'apogee':{'clusters' : read_caldata,    # Sample of clusters
                    'OCs': read_caldata,          # Sample of open clusters
                    'GCs': read_caldata,          # Sample of globular clusters
                    'red_clump' : apread.rcsample,# Sample of red clump star
                    'red_giant' : rgsample,       # Sample of red giant star
                    'syn': get_synthetic
                    }
}

# List of accepted keys to do slice in
keyList = ['RA','DEC','GLON','GLAT','TEFF','LOGG','TEFF_ERR','LOGG_ERR',
            'AL_H','CA_H','C_H','FE_H','K_H','MG_H','MN_H','NA_H','NI_H',
            'N_H','O_H','SI_H','S_H','TI_H','V_H','CLUSTER','MEANFIB','SIGFIB']
keyList.sort()

# List of accepted keys for upper and lower limits
_upperKeys = ['max','m','Max','Maximum','maximum','']
_lowerKeys = ['min','m','Min','Minimum','minimum','']

class starSample(object):
    """
    Gets properties of a sample of stars given a key that defines the
    read function.

    """
    def __init__(self,dataSource,sampleType,ask=True,datadict=None):
        """
        Get properties for all stars that match the sample type

        sampleType:   designator of the sample type - must be a key in readfn
                      and independentVariables in data.py

        """
        if sampleType == 'syn':
            self._sampleType = sampleType
            self._dataSource = dataSource
            self.DR = '0'
            if not isinstance(datadict,dict):
                print('Initialized empty star sample object, call get_synthetic(), passing the name of this object as the first argument')
            elif isinstance(datadict,dict):
                get_synthetic(self,datadict)


        elif sampleType != 'syn':
            self._dataSource = dataSource
            if self._dataSource == 'apogee':
                if ask:
                    self.DR = input('Which data release? (Enter for 12): ')
                    if self.DR=='':
                        self.DR='12'
                if not ask:
                    self.DR = '12'
                if self.DR=='12':
                    os.environ['RESULTS_VERS']='v603'
                    change_dr('12')
                if self.DR=='13':
                    os.environ['RESULTS_VERS']='l30e.2'
                    change_dr('13')
                os.system('echo RESULTS_VERS $RESULTS_VERS')
                change_dr(self.DR)
            self._sampleType = sampleType
            self._getProperties()

    def _getProperties(self):
        """
        Get properties of all possible stars to be used.
        """
        if self.DR:
            self.data = readfn[self._dataSource][self._sampleType]()
            if self.DR=='12':
                fib = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','data','DR12_supplement','fiberinfo.npy'))
                if self._sampleType=='clusters':
                    notmissing = (np.array([i for i in range(len(self.data['APOGEE_ID'])) if self.data['APOGEE_ID'][i] in fib['APOGEE_ID']]),)
                else:
                    notmissing = (np.arange(0,len(self.data)),)
                import numpy.lib.recfunctions as rfunc
                self.data = rfunc.append_fields(self.data,('MEANFIB','SIGFIB'),data=(np.zeros(len(self.data)),np.zeros(len(self.data))),dtypes=('f4','f4'),usemask=False)
                meanfib =  dict(zip(fib['APOGEE_ID'],fib['MEANFIB']))
                sigfib = dict(zip(fib['APOGEE_ID'],fib['SIGFIB']))
                self.data['MEANFIB'][notmissing] = np.array([meanfib[apoid] for apoid in self.data['APOGEE_ID'][notmissing]])
                self.data['SIGFIB'][notmissing] = np.array([sigfib[apoid] for apoid in self.data['APOGEE_ID'][notmissing]])
        print('properties ',dir(self))

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
        self.c_h = np.ma.masked_array(np.zeros((len(stardata)),
                                                dtype=float))
        self.n_h = np.ma.masked_array(np.zeros((len(stardata)),
                                                dtype=float))
        self.o_h = np.ma.masked_array(np.zeros((len(stardata)),
                                                dtype=float))
        self.fib = np.ma.masked_array(np.zeros((len(stardata)),
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
        missing = 0
        # Fill arrays for each star
        print(stardata.dtype)
        for star in tqdm(range(len(stardata)),desc='read star data'):
            LOC = stardata[star]['LOCATION_ID']
            APO = stardata[star]['APOGEE_ID']
            TEFF = stardata[star]['TEFF']
            LOGG = stardata[star]['LOGG']
            FE_H = stardata[star]['FE_H']
            if self.DR=='12':
                C_H = stardata[star]['C_H']
                N_H = stardata[star]['N_H']
                O_H = stardata[star]['O_H']
            elif self.DR=='13':
                C_H = stardata[star]['C_FE']
                N_H = stardata[star]['N_FE']
                O_H = stardata[star]['O_FE']
            FIB = stardata[star]['MEANFIB']

            # Fit variables
            self.teff[star] = np.ma.masked_array(TEFF)
            self.logg[star] = np.ma.masked_array(LOGG)
            self.fe_h[star] = np.ma.masked_array(FE_H)
            self.c_h[star] = np.ma.masked_array(C_H)
            self.n_h[star] = np.ma.masked_array(N_H)
            self.o_h[star] = np.ma.masked_array(O_H)
            self.fib[star] = np.ma.masked_array(FIB)

            # Spectral data
            try:
                self.spectra[star] = apread.aspcapStar(LOC,APO,ext=1,
                                                       header=False,dr=self.DR,
                                                       aspcapWavegrid=True)
                self.spectra_errs[star] = apread.aspcapStar(LOC,APO,ext=2,
                                                            header=False,
                                                            dr=self.DR,
                                                            aspcapWavegrid=True)
                self._bitmasks[star] = apread.apStar(LOC,APO,ext=3,
                                                     header=False,dr=self.DR,
                                                     aspcapWavegrid=True)[1]
            except IOError:
                print('Star {0} missing '.format(star))
                self.spectra[star] = np.zeros(aspcappix)
                self.spectra_errs[star] = np.ones(aspcappix)
                self._bitmasks[star] = np.ones(aspcappix).astype(np.int16)
                missing +=1

            if LOGG<-1000 or TEFF<-1000 or FE_H<-1000 or self.data[star]['SIGFIB'] < 0 or self.data[star]['MEANFIB'] < 0:
                self._bitmasks[star] = np.ones(aspcappix).astype(np.int16)

        print('Total {0} of {1} stars missing'.format(missing,len(stardata)))


    def show_sample_coverage(self,coords=True,phi_ind='RC_GALPHI',r_ind='RC_GALR',z_ind='RC_GALZ'):
        """
        Plots the sample in Galacto-centric cylindrical coordinates.

        """
        # Start figure
        plt.figure(figsize=(10,5.5))
        # Set up Cartesian axes
        car = plt.subplot(121)
        # Set up polar axes
        pol = plt.subplot(122,projection='polar')

        if coords:
            # Find location data
            phi = self.data[phi_ind]
            r = self.data[r_ind]
            z = self.data[z_ind]

        # Plot data
        car.plot(r,z,'ko',markersize=2,alpha=0.2)
        pol.plot(phi,r,'ko',markersize=2,alpha=0.2)
        # Reorient polar plot to match convention
        pol.set_theta_direction(-1)

        # Constrain plot limits and set labels
        car.set_xlim(min(r),max(r))
        car.set_ylim(min(z),max(z))
        car.set_xlabel('R (kpc)')
        car.set_ylabel('z (kpc)')
        pol.set_rlabel_position(135)
        pol.set_rlim(min(r),max(r))
        pol.set_xticks([])
        plt.subplots_adjust(wspace=0.05)

    def plotHistogram(self,array,title = '',xlabel = '',norm=True,
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
        plt.figure(figsize=(10,8))
        hist,binEdges = np.histogram(array,**kwargs)
        if norm:
            area = np.sum(hist*(binEdges[1]-binEdges[0]))
            barlist = plt.bar(binEdges[:-1],hist/area,width = binEdges[1]-binEdges[0])
        elif not norm:
            barlist = plt.bar(binEdges[:-1],hist,width = binEdges[1]-binEdges[0])
        colours = plt.get_cmap('plasma')(np.linspace(0, 0.85, len(barlist)))
        for bar in range(len(barlist)):
            barlist[bar].set_color(colours[bar])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if saveName:
            plt.savefig('plots/'+saveName+'.png')
            plt.close()




class makeFilter(starSample):
    """
    Contains functions to create a filter and associated directory
    name for a starSample.
    """
    def __init__(self,dataSource,sampleType,ask=True,datadict=None,datadir='.',func=None,file=None):
        """
        Sets up filter_function.py file to contain the appropriate function
        and puts the save directory name in the docstring of the function.

        sampleType:   designator of the sample type - must be a key in readfn
                      and independentVariables in data.py
        ask:          if True, function asks for user input to make
                      filter_function.py, if False, uses existing
                      filter_function.py

        """
        starSample.__init__(self,dataSource,sampleType,ask=ask,datadict=datadict)
        self.datadir=datadir
        if ask:
            self.done = False
            print('Type done at any prompt when finished')
            # Start name and condition string
            self.name = self.datadir+'/'+self._sampleType+'_'+str(self.DR)
            self.condition = ''
            # Ask for new key conditions until the user signals done
            while not self.done:
                self.done = self._sampleInfo()
            # Check that the user set conditions.
            # If conditions not set, recursively call init
            if self.condition == '':
                print('No conditions set')
                self.__init__(dataSource,sampleType,ask=True)
            # When conditions set, trim trailing ampersand
            self.condition = self.condition[:-2]
            # Write the new filter function to file
            f = open('filter_function.py','w')
            f.write(self._basicStructure()+self.condition)
            f.close()
        elif not ask:
            if callable(func):
                starFilter=func
                self.name = starFilter.__doc__.split('\n')[-2]
                self.name = self.name.split('\t')[-1]
                self.name = self.name.strip()
                f = open('filter_function.py','w')
                functext = ''.join(inspect.getsourcelines(starFilter)[0])
                f.write('import numpy as np\n\n'+functext)
                f.close()
            elif not callable(func):
                # Import existing filter function. If function doesn't exist,
                # recursively call init
                if isinstance(file,str):
                    f = open(file,'r')
                    filter_text = f.readlines()
                    f.close()
                    f = open('filter_function.py','w')
                    f.write(filter_text)
                    f.close()
                else:
                    try:
                        import filter_function
                        reload(filter_function)
                        from filter_function import starFilter
                        self.name = starFilter.__doc__.split('\n')[-2]
                        self.name = self.name.split('\t')[-1]
                    except ImportError:
                        print('filter_function.py does not contain the required starFilter function.')
                        self.__init__(dataSource,sampleType,ask=True)
        self.getDirectory()
        self.filterCopy()

    def _basicStructure(self):
        """
        Returns the basic form of filter_function.py

        """
        return 'import numpy as np\n\ndef starFilter(data):\n\t"""\n\t{0}\n\t"""\n\treturn'.format(self.name)

    def _sampleInfo(self):
        """
        Retrieves information about the sample from the user.

        """
        key = input('Data key: ')
        # Check if key is accepted
        if key in keyList:
            self.name+='_'+key
            # Get info for this key
            match = self._match(key)
            if match[0]=='done':
                return True
            elif match[0]=='a':
                self.name+='_fullsample'
                self.condition = 'np.where(data)'
                return True
            elif match[0]=='m':
                # Add string form of the matching condition and
                # update the name
                self.name+='_match'+match[1]
                andor = input('And/or? ')
                if andor == 'and' or andor=='a' or andor=='&':
                    self.condition += ' (data[\'{0}\'] == "{1}") &'.format(key,match[1])
                    return False
                elif andor == 'or' or andor=='o' or andor=='|':
                    self.condition += ' (data[\'{0}\'] == "{1}") |'.format(key,match[1])
                    return False
                elif andor == 'done':
                    self.condition += ' (data[\'{0}\'] == "{1}") &'.format(key,match[1])
                    return True
                else:
                    print('Invalid choice of "and" or "or", using "or" by default')
                    self.condition += ' (data[\'{0}\'] == "{1}") |'.format(key,match[1])
                    return False
            elif match[0]=='s':
                # Add string form of the slicing condition and
                # update the name
                self.name+='_up'+str(match[1])+'_lo'+str(match[2])
                andor = input('And/or? ')
                if andor == 'and' or andor=='a' or andor=='&':
                    self.condition += ' (data[\'{0}\'] < {1}) & (data[\'{0}\'] > {2}) &'.format(key,match[1],match[2])
                    return False
                elif andor == 'or' or andor=='o' or andor=='|':
                    self.condition += ' ((data[\'{0}\'] < {1}) & (data[\'{0}\'] > {2})) |'.format(key,match[1],match[2])
                    return False
                elif andor =='done':
                    self.condition += ' ((data[\'{0}\'] < {1}) & (data[\'{0}\'] > {2})) &'.format(key,match[1],match[2])
                    return True
                else:
                    print('Invalid choice of "and" or "or", using "or" by default')
                    self.condition += ' ((data[\'{0}\'] < {1}) & (data[\'{0}\'] > {2})) |'.format(key,match[1],match[2])
                    return False
        # If key not accepted, make recursive call
        elif key not in keyList and key != 'done':
            print('Got a bad key. Try choosing one of ',keyList)
            result = self._sampleInfo()
            return result
        # If done condition, exit
        elif key == 'done':
            print('Done getting filter information')
            return True


    def _match(self,key):
        """
        Returns user-generated conditions to match or slice in a given key.

        key:   label of property of the data set

        """
        # Check whether we will match to key or slice in its range
        match = input('Default is full range. Match or slice? ').strip()

        if match == 'match' or match == 'm' or match == 'Match':
            m = input('Match value: ')
            if m=='done':
                print('Done getting filter information')
                return 'done',None
            # Check if match value has at least one star,
            # if not call _match recursively
            elif m!='done' and m in self.data[key]:
                return 'm',m
            elif m not in self.data[key]:
                print('No match for this key. Try choosing one of ',np.unique(self.data[key]))
                self._match(key)

        elif match == 'slice' or match == 's' or match == 'Slice':
            # Get limits of slice
            upperLimit = input('Upper limit (Enter for maximum): ')
            lowerLimit = input('Lower limit (Enter for minimum): ')
            if upperLimit == 'done' or lowerLimit == 'done':
                print('Done getting filter information')
                return 'done',None
            elif upperLimit != 'done' and lowerLimit != 'done':
                if upperLimit == 'max' or upperLimit == 'm' or upperLimit == '':
                    upperLimit = np.max(self.data[key])
                if lowerLimit == 'min' or lowerLimit == 'm' or lowerLimit == '':
                    lowerLimit = np.min(self.data[key])
                # Check limits are good - if not, call _match recursively
                try:
                    if float(upperLimit) <= float(lowerLimit):
                        print('Limits are the same or are in the wrong order. Try again.')
                        self._match(key)
                    elif float(upperLimit) > float(lowerLimit):
                        print('Found good limits')
                        return 's',float(upperLimit),float(lowerLimit)
                except ValueError as e:
                    print('Please enter floats for the limits')
                    self._match(key)

        # Option to use the entire sample
        elif match == 'all' or match == 'a' or match == 'All':
            return 'a',None

        # Exit filter finding
        elif match == 'done':
            print('Done getting filter information')
            return 'done',None

        # Invalid entry condition
        else:
            print('Invalid choice, please type match, slice or all')
            result = self._match(key)
            return result

    def getDirectory(self):
        """
        Create directory to store results for given filter.

        """
        if not os.path.isdir(self.name):
            os.system('mkdir -p {0}/'.format(self.name))
        return

    def directoryClean(self):
        """
        Removes all files from a specified directory.

        """
        os.system('rm -rf {0}/*.npy'.format(self.name))

    def filterCopy(self):
        """
        Copies filter function to data directory.
        """
        os.system('cp filter_function.py {0}/'.format(self.name))

class subStarSample(makeFilter):
    """
    Given a filter function, defines a subsample of the total sample of stars.

    """
    def __init__(self,dataSource,sampleType,ask=True,datadict=None,datadir='.',func=None):
        """
        Create a subsample according to a starFilter function

        sampleType:   designator of the sample type - must be a key in readfn
                      and independentVariables in data.py
        ask:          if True, function asks for user input to make
                      filter_function.py, if False, uses existing
                      filter_function.py

        """
        # Create starFilter
        makeFilter.__init__(self,dataSource,sampleType,ask=ask,datadict=datadict,datadir=datadir,func=func)
        import filter_function
        reload(filter_function)
        from filter_function import starFilter
        # Find stars that satisfy starFilter and cut data accordingly
        change_dr(self.DR)
        self._matchingStars = starFilter(self.data)
        self.matchingData = self.data[self._matchingStars]
        #self.numberStars = len(self.matchingData)
        if self._sampleType != 'syn':
            self.checkArrays()

    def numberStars(self):
        return len(self.matchingData)



    def checkArrays(self):
        """
        Check if input data has already been saved as arrays.
        If not, create them.

        """

        fnames = np.array([self.name+'/teff.npy',
                           self.name+'/logg.npy',
                           self.name+'/fe_h.npy',
                           self.name+'/c_h.npy',
                           self.name+'/n_h.npy',
                           self.name+'/o_h.npy',
                           self.name+'/fib.npy',
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
            self.c_h = np.load(self.name+'/c_h.npy')
            self.n_h = np.load(self.name+'/n_h.npy')
            self.o_h = np.load(self.name+'/o_h.npy')
            self.fib = np.load(self.name+'/fib.npy')
            self.spectra = np.ma.masked_array(np.load(self.name+'/spectra.npy'))
            self.spectra_errs = np.ma.masked_array(np.load(self.name+'/spectra_errs.npy'))
            self._bitmasks = np.load(self.name+'/bitmasks.npy')
        # If any file is missing, generate arrays and write to file
        elif not fexist:
            self.makeArrays(self.matchingData)
            np.save(self.name+'/teff.npy',self.teff.data)
            np.save(self.name+'/logg.npy',self.logg.data)
            np.save(self.name+'/fe_h.npy',self.fe_h.data)
            np.save(self.name+'/c_h.npy',self.c_h.data)
            np.save(self.name+'/n_h.npy',self.n_h.data)
            np.save(self.name+'/o_h.npy',self.o_h.data)
            np.save(self.name+'/fib.npy',self.fib.data)
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

    def imshow(self,plotData,saveName=None,title = '',xlabel='pixels',ylabel='stars',zlabel='',**kwargs):
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
        plt.colorbar(label=zlabel)
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
