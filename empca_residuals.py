import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import statsmodels.nonparametric.smoothers_lowess as sm
import access_spectrum as acs
from empca import empca,MAD,meanMed
from mask_data import mask,maskFilter
from data_access import *
import access_spectrum as acs 
import os
from galpy.util import multi as ml

testmode = False

font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  20
}

matplotlib.rc('font',**font)

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


def getsmallEMPCAarrays(model):
     """                                                                                 
     Read out arrays                                                                     
     """
     if model.savename:
         arc = np.load('{0}_data.npz'.format(model.savename))
        
         model.eigval = arc['eigval']
         model.eigvec = np.ma.masked_array(arc['eigvec'],
                                          mask=arc['eigvecmask'])
         model.coeff = arc['coeff']
         
class smallEMPCA(object):
    """
    
    Class to contain crucial EMPCA-related objects.

    """
    def __init__(self,model,correction=None,savename=None):
        """
        Set all relevant data for the EMPCA.

        model:        EMPCA model object
        correction:   information about correction used on data
        """
        self.R2Array = model.R2Array
        self.R2noise = model.R2noise
        self.Vnoise = model.Vnoise
        self.Vdata = model.Vdata
        self.eigvec = model.eigvec
        self.coeff = model.coeff
        self.correction = correction
        self.eigval = model.eigvals
        self.savename= savename
        self.cleararrays()

    def cleararrays(self):
        """                                                                    
        Clear memory allocation for bigger arrays by writing them to file if 
        self.savename is set                                                            
        """
        if self.savename:
            np.savez_compressed('{0}_data.npz'.format(self.savename),
                                eigval=self.eigval,eigvec=self.eigvec.data,
                                eigvecmask = self.eigvec.mask,
                                coeff=self.coeff)
        del self.eigval
        del self.eigvec
        del self.coeff

class empca_residuals(mask):
    """
    Contains functions to find polynomial fits.
    
    """
    def __init__(self,dataSource,sampleType,maskFilter,ask=True,datadir='.',
                 subsamples=1,badcombpixmask=4351,minSNR=50,degree=2,
                 varfuncs = [np.ma.var,MAD,meanMed],nvecs=5,**kwargs):
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
        mask.__init__(self,dataSource,sampleType,maskFilter,ask=ask,
                      minSNR=minSNR,datadir=datadir,
                      badcombpixmask=badcombpixmask)
        self.degree = degree
        self.subsamples = subsamples
        self.polynomial = PolynomialFeatures(degree=degree)
        self.testM = self.makeMatrix(0)
        self.varfuncs = varfuncs
        self.nvecs = nvecs
        if testmode:
            self.division = False
            self.jackknife(**kwargs)

    def sample_wrapper(self,i):
        """
        Runs many subsamples of full sample choice.

        If self.division is True, define the subsample as 
        where the randomly assigned sample numbers match i
        If self.division is False, define the subsamples as 
        where the randomly assigned sample numbers do not match i
        
        """
        # Select ith subsample and update arrays
        if self.division:
            self.matchingData = self.filterData[self.inds==i]
            self.teff = self.originalteff[self.inds==i]
            self.logg = self.originallogg[self.inds==i]
            self.fe_h = self.originalfe_h[self.inds==i]
            self.spectra = self.originalspectra[self.inds==i]
            self.spectra_errs = self.originalspectra_errs[self.inds==i]
            self._bitmasks = self.originalbitmasks[self.inds==i]
            self._maskHere = self.originalmaskHere[self.inds==i]
        if not self.division:
            self.matchingData = self.filterData[self.inds!=i]
            self.teff = self.originalteff[self.inds!=i]
            self.logg = self.originallogg[self.inds!=i]
            self.fe_h = self.originalfe_h[self.inds!=i]
            self.spectra = self.originalspectra[self.inds!=i]
            self.spectra_errs = self.originalspectra_errs[self.inds!=i]
            self._bitmasks = self.originalbitmasks[self.inds!=i]
            self._maskHere = self.originalmaskHere[self.inds!=i]
        # Update mask
        self.applyMask()
        # Update name
        self.name = '{0}/seed{1}_subsample{2}of{3}/'.format(self.originalname,
                                                            self.seed,i+1,
                                                            self.subsamples)
        # Create directory and solve fits
        self.getDirectory()
        self.findResiduals()
        # Create output arrays to hold EMPCA results for each variance function
        self.R2As = np.zeros((len(self.varfuncs),self.nvecs+1))
        self.R2ns = np.zeros((len(self.varfuncs)))
        self.cvcs = np.zeros((len(self.varfuncs)))
        self.labs = np.zeros((len(self.varfuncs)),dtype='S100')
        # Store information about which sample you're on
        self.samplenum = i+1
        # Call EMPCA solver for all variance functions in parallel
        stat = ml.parallel_map(self.EMPCA_wrapper,range(len(self.varfuncs)))
        # Unpack results of running in parallel and store
        for s in range(len(stat)):
            R2A,R2n,cvc,lab = stat[s]
            self.R2As[s] = R2A
            self.R2ns[s] = R2n
            self.cvcs[s] = cvc
            self.labs[s] = lab
        # Clear arrays from memory
        del self.matchingData 
        del self.filterData
        del self.teff
        del self.originalteff
        del self.logg
        del self.originallogg
        del self.fe_h
        del self.originalfe_h
        del self.spectra 
        del self.originalspectra
        del self.spectra_errs 
        del self.originalspectra_errs
        del self._bitmasks
        del self.originalbitmasks
        del self._maskHere 
        del self.originalmaskHere
        return self

    def EMPCA_wrapper(self,v):
        """
        Run EMPCA for vth variance function and compute statistics.
        """
        # run EMPCA
        self.pixelEMPCA(varfunc=self.varfuncs[v],nvecs=self.nvecs,
                        savename='eig{0}_minSNR{1}_corrNone_{2}.pkl'.format(self.nvecs,self.minSNR,self.varfuncs[v].__name__))
        # Find R^2 values
        R2A = self.empcaModelWeight.R2Array
        R2n = self.empcaModelWeight.R2noise
        # Find where R^2 intersects R^2_noise
        crossvec = np.where(self.empcaModelWeight.R2Array > self.empcaModelWeight.R2noise)[0]
        if crossvec.shape[0] == 0:
            cvc = -1
        elif crossvec.shape[0] != 0:
            cvc = crossvec[0]-1
            if cvc <0:
                cvc=0
        # Create label for sample
        lab = 'subsamp {0}, {1} stars, func {2} - {3} vec'.format(self.samplenum,self.numberStars(),self.varfuncs[v].__name__,cvc)
        return (R2A,R2n,cvc,lab)

    def samplesplit(self,division=False,seed=None,fullsamp=True,maxsamp=5):
        """                                                                     
        Take self.subsamples random subsamples of the original data set and
        run EMPCA.

        Takes kwargs for pixelEMPCA

        Creates a plot comparing R^2 statistics for the subsamples.

        """
        self.division=division
        # If no subsamples, just run regular EMCPA
        if self.subsamples==1:
            self.findResiduals()
            R2Arrays = np.zeros((len(self.varfuncs),self.nvecs+1))
            R2noises = np.zeros((len(self.varfuncs)))
            crossvecs = np.zeros((len(self.varfuncs)))
            labels = np.zeros(len(self.varfuncs),dtype='S100')
            self.samplenum=1
            # Call EMPCA solver for all variance functions in parallel
            stat = ml.parallel_map(self.EMPCA_wrapper,range(len(self.varfuncs)))
            # Unpack results of running in parallel and store
            for s in range(len(stat)):
                R2A,R2n,cvc,lab = stat[s]
                R2Arrays[s] = R2A
                R2noises[s] = R2n
                crossvecs[s] = cvc
                labels[s] = lab
            
        # If subsamples, run EMPCA on many subsamples
        elif self.subsamples!=1:
            # Pick seed to initialize randomization for reproducibility
            if not seed:
                self.seed = np.random.randint(0,100)
            elif seed:
                self.seed=seed
            np.random.seed(self.seed)
            # Make copies of original data to use for slicing
            self.filterData = np.copy(self.matchingData)
            self.originalteff = np.ma.copy(self.teff)
            self.originallogg = np.ma.copy(self.logg)
            self.originalfe_h = np.ma.copy(self.fe_h)
            self.originalspectra = np.ma.copy(self.spectra)
            self.originalspectra_errs = np.ma.copy(self.spectra_errs)
            self.originalbitmasks = np.copy(self._bitmasks)
            self.originalmaskHere = np.copy(self._maskHere)
            self.originalname = np.copy(self.name)
            # Initialize array that assigns each star to a group from 0
            # To self.subsamples-1
            self.inds = np.zeros((self.numberStars()))-1
            # Number of stars in each subsample
            subnum = self.numberStars()/self.subsamples 
            # Randomly choose stars to belong to each subsample
            for i in range(self.subsamples):
                group = np.random.choice(np.where(self.inds==-1)[0],size=subnum,replace=False)
                self.inds[group] = i
            # Distribute leftover stars one at a time to each subsample
            leftovers = [i for i in range(self.numberStars()) if self.inds[i]==-1]
            if leftovers != []:
                k = 0
                for i in leftovers:
                    self.inds[i] = k
                    k+=1
            # Create arrays to hold R^2 statistics and their labels
            if fullsamp:
                self.sampnum = self.subsamples+1
            elif not fullsamp:
                self.sampnum = self.subsamples
            R2Arrays = np.zeros((len(self.varfuncs)*(self.sampnum),
                                 self.nvecs+1))
            R2noises = np.zeros((len(self.varfuncs)*(self.sampnum)))
            crossvecs = np.zeros((len(self.varfuncs)*(self.sampnum)))
            labels = np.zeros((len(self.varfuncs)*(self.sampnum)),dtype='S200')
            # Run all samples in parallel but serialize if too large
            if self.sampnum <=maxsamp:
                stats = ml.parallel_map(self.sample_wrapper, range(self.sampnum))
            elif self.sampnum >maxsamp:
                sample = 0
                stats = []
                number_sets = self.sampnum/maxsamp + int(self.sampnum % maxsamp > 0)
                if self.sampnum/maxsamp == number_sets:
                    for i in range(number_sets):
                        ss = ml.parallel_map(self.sample_wrapper, range(sample,sample+maxsamp))
                        sample += maxsamp
                        stats.append(ss)
                elif self.sampnum/maxsamp == number_sets:
                    for i in range(number_sets):
                        if i < number_sets-1:
                            ss = ml.parallel_map(self.sample_wrapper, range(sample,sample+maxsamp))
                            sample += maxsamp
                            stats.append(ss)
                        if i == number_set-1:
                            ss = ml.parallel_map(self.sample_wrapper, range(sample,sample+self.sampnum % maxsamp))
                            sample += self.sampnum% maxsamp
                            stats.append(ss)
                stats = [item for sublist in stats for item in sublist]
            # Unpack information from parallel runs into appropriate arrays
            k = 0
            for s in range(len(stats)):
                model = stats[s]
                R2Arrays[k:k+len(self.varfuncs)] = model.R2As
                R2noises[k:k+len(self.varfuncs)] = model.R2ns
                crossvecs[k:k+len(self.varfuncs)] = model.cvcs
                labels[k:k+len(self.varfuncs)] = model.labs
                k+=len(self.varfuncs)
            print R2Arrays, R2noises, crossvecs, labels, 
            # Restore original arrays
            self.matchingData = self.filterData
            self.teff = self.originalteff
            self.logg = self.originallogg
            self.fe_h = self.originalfe_h
            self.spectra = self.originalspectra
            self.spectra_errs = self.originalspectra_errs
            self._bitmasks = self.originalbitmasks
            self._maskHere = self.originalmaskHere
            self.name = str(self.originalname)
            # Update mask
            self.applyMask()
            # Calculate uncertainty on number of eigenvectors.
            sort = self.func_sort(R2Arrays,R2noises,crossvecs,labels)
            labs = sort[3]
            cvecs = sort[2]
            start = 0
            for v in range(len(self.varfuncs)):
                num = len(cvecs)/len(self.varfuncs)
                lab = labs[start:start+num]
                cvec = cvecs[start:start+num]
                start+=num
                safe = np.array([i for i in range(len(lab)) if 'subsamp{0}'.format(self.subsamples+1) not in lab[i]])
                cvec = cvec[safe]
                avgvec = np.median(cvec)
                if not self.division:
                    varvec = ((len(cvec)-1.)/float(len(cvec)))*np.sum((cvec-avgvec)**2)
                elif self.division:
                    varvec = np.var(cvec)
                print '{0} +/- {1}'.format(avgvec,np.sqrt(varvec))
                self.numeigvec = avgvec
                self.numeigvec_std = np.sqrt(varvec)
                numeigvec_file = np.array([self.numeigvec,self.numeigvec_std])
                numeigvec_file.tofile('{0}/subsamples{1}_{2}_seed{3}_numeigvec.npy'.format(self.name,self.subsamples,self.varfuncs[v].__name__,self.seed))
        # Move full sample analysis to parent directory
        if fullsamp:
            os.system('mv {0}/seed{1}_subsample{2}of{3}/* {4}'.format(self.name,self.seed,self.subsamples+1,self.subsamples,self.name))
            os.system('rmdir {0}/seed{1}_subsample{2}of{3}/'.format(self.name,self.seed,self.subsamples+1,self.subsamples))
        # Make plots sorting by function
        self.R2compare(R2Arrays,R2noises,crossvecs,labels,funcsort=True)
        self.R2compare(R2Arrays,R2noises,crossvecs,labels,funcsort=False)

            

    def func_sort(self,R2Arrays,R2noises,crossvecs,labels):
        newR2 = np.zeros(R2Arrays.shape)
        newR2n = np.zeros(R2noises.shape)
        newvec = np.zeros(crossvecs.shape)
        newlab = []
        k = 0
        for i in range(len(self.varfuncs)):
            print k,newR2.shape, R2Arrays.shape
            newR2[k:k+self.sampnum] = R2Arrays[i::len(self.varfuncs)]
            newR2n[k:k+self.sampnum] = R2noises[i::len(self.varfuncs)]
            newvec[k:k+self.sampnum] = crossvecs[i::len(self.varfuncs)\
                                                  ]
            newlab.append(labels[i::len(self.varfuncs)])
            k += self.sampnum
        return newR2,newR2n,newvec,[item for sublist in newlab for item in sublist]
    
    def R2compare(self,R2Arrays,R2noises,crossvecs,labels,funcsort=True):
        """
        Make plots comparing R2 values for different samples and different 
        functions. If there are less than 10 subsamples (including each
        variance function as a different sample), this plots a 1D comparison
        of R^2 values as a function of eigenvector number. Regardless of the
        number of subsamples, this plots a 2D comparison of R^2.

        R2Arrays:    An array of R2 values with size number of samples by 
                     number of eigenvectors
        R2noises:    An array of R2noise values with size number of samples
        crossvecs:   An array of the location of the intersection of R2 and
                     R2noise with size number of samples
        labels:      Labels for each subsample
        funcsort:    Keyword to sort input to group similar functions instead
                     of similar samples

        Saves 1 or 2 figures.
        """
        # If sorting by function instead of sample, slice and reorient arrays 
        # accordingly
        if funcsort:
            R2Arrays,R2noises,crossvecs,labels = self.func_sort(R2Arrays,R2noises,crossvecs,labels)
        # Get colours for line plot
        colors = plt.get_cmap('plasma')(np.linspace(0,0.85,len(labels)))
        # If there aren't too many lines, make a 1D line plot of R2
        if (self.subsamples+1)*len(self.varfuncs) < 10:
            plt.figure(figsize=(10,8))
            for i in range(len(labels)):
                plt.plot(R2Arrays[i],lw=4,color=colors[i],label=labels[i])
                plt.axhline(R2noises[i],lw=3,ls='--',color=colors[i])
            plt.ylim(0,1)
            plt.xlim(-1,len(R2Arrays[0]))
            plt.ylabel('$R^2$',fontsize=20)
            plt.xlabel('n')
            legend = plt.legend(loc='best',fontsize = 13)
            if hasattr(self,'seed'):
                plt.savefig('{0}/seed{1}_R2comp.png'.format(self.name,self.seed))
            elif not hasattr(self,'seed'):
                plt.savefig('{0}/R2comp.png'.format(self.name))
        # Make a 2D plot of R2
        plt.figure(figsize=(10,8))
        plt.subplot(111)
        plt.imshow(R2Arrays,cmap = 'viridis',interpolation = 'nearest',
                   aspect = R2Arrays.shape[1]/float(R2Arrays.shape[0]),
                   vmin=0,vmax=1.0)
        # Plot lines to split up samples or function types, and vertical lines
        # to mark cross over points between R2 and R2noise
        k = 0
        for i in range(len(labels)):
            # Find bounds on the scale of the plotting area (0-1)
            ymin = i*(1./len(labels))
            ymax = ymin + (1./len(labels))
            # Plot marker of where R2 crosses R2noise
            plt.axvline(crossvecs[i],ymin=ymin,ymax=ymax,color='w',lw=2)
            # Plot horizontal lines to guide the eye
            # If not sorted, mark bounds of samples
            if not funcsort:
                if i == k*len(self.varfuncs):
                    if k!=0:
                        plt.axhline(i-0.5,color='w',lw=2,ls='--')
                    k+=1
            # If sorted, mark bounds of functions
            elif funcsort:
                if i == k*(self.subsamples+1):
                    if k!=0:
                        plt.axhline(i-0.5,color='w',lw=2,ls='--')
                    k+=1
        # Find shorter labels for the 2D plots
        shortlabels = [i[:-10] for i in labels]
        # If there are too many labels, reduce them for readability
        if (self.sampnum*len(self.varfuncs)) > 10:
            few = int(np.ceil(np.log10(self.sampnum)))
        else:
            few = 1
        # Plot ylables
        plt.yticks(np.arange(len(labels))[::few],shortlabels[::-1][::few],
                   fontsize=12)
        # Constrain x-axis since adding axhline/axvline can make the axes stretch
        plt.xlim(-0.5,self.nvecs+0.5)
        plt.xlabel('$n$')
        plt.colorbar(label='$R^2$')
        plt.tight_layout()
        # Save the plot
        if hasattr(self,'seed'):
            if funcsort:
                plt.savefig('{0}/seed{1}_fsort_2D_R2comp.png'.format(self.name,self.seed))
            elif not funcsort:
                plt.savefig('{0}/seed{1}_2D_R2comp.png'.format(self.name,self.seed))
        elif not hasattr(self,'seed'):
            if funcsort:
                plt.savefig('{0}/fsort_2D_R2comp.png'.format(self.name))
            elif not funcsort:
                plt.savefig('{0}/2D_R2comp.png'.format(self.name))

    def makeMatrix(self,pixel,matrix='default'):
        """
        Find independent variable matrix
        
        pixel:   pixel to use, informs the mask on the matrix
        
        Returns the matrix
        """
        if matrix=='default':
            matrix=self._sampleType
        # Find the number of unmasked stars at this pixel
        numberStars = len(self.spectra[:,pixel][self.unmasked[:,pixel]])
        # Create basic independent variable array
        indeps = np.zeros((numberStars,
                           len(independentVariables[self._dataSource][matrix])))

        for i in range(len(independentVariables[self._dataSource][matrix])):
            variable = independentVariables[self._dataSource][matrix][i]
            indep = self.keywordMap[variable][self.unmasked[:,pixel]]
            indeps[:,i] = indep-np.median(indep)
        # use polynomial to produce matrix with all necessary columns
        return np.matrix(self.polynomial.fit_transform(indeps))

    def findFit(self,pixel,eigcheck=False,givencoeffs=[],matrix='default'):
        """
        Fits polynomial to all spectra at a given pixel, weighted by spectra 
        uncertainties.

        pixel:   pixel at which to perform fit

        Return polynomial values and coefficients

        """
        # find matrix for polynomial of independent values
        indeps = self.makeMatrix(pixel,matrix=matrix)
        self.numparams = indeps.shape[1]

        if givencoeffs == []:
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
        elif givencoeffs != []:
            coeffs,coeff_errs = givencoeffs
            bestFit = indeps*coeffs
        return bestFit,coeffs.T,coeff_errs

    def multiFit(self,minStarNum='default',eigcheck=False,coeffs=None,matrix='default'):
        """
        Loop over all pixels and find fit. Mask where there aren't enough 
        stars to fit.

        minStarNum:   (optional) number of stars required to perform fit 
                      (default:'default' which sets minStarNum to the number 
                       of fit parameters plus one)
        
        """
        # create sample matrix to confirm the number of parameters
        self.testM = self.makeMatrix(0,matrix=matrix)
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
            
        if not coeffs:
            
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
                    fitSpectrum,coefficients,coefficient_uncertainty = self.findFit(pixel,eigcheck,matrix=matrix)
                    self.fitSpectra[:,pixel][self.unmasked[:,pixel]] = fitSpectrum
                    self.fitCoeffs[pixel] = coefficients
                    self.fitCoeffErrs[pixel] = coefficient_uncertainty
        elif coeffs:
            fmask = np.load(self.name+'/fitcoeffmask.npy')
            self.fitCoeffs = np.ma.masked_array(np.load(self.name+'/fitcoeffs.npy'),mask=fmask)
            self.fitCoeffErrs = np.ma.masked_array(np.load(self.name+'/fitcoefferrs.npy'),mask=fmask)
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
                    fitSpectrum,coefficients,coefficient_uncertainty = self.findFit(pixel,eigcheck,
                                                                                    givencoeffs = [self.fitCoeffs[pixel],self.fitCoeffErrs[pixel]],
                                                                                    matrix=matrix)
                    self.fitSpectra[:,pixel][self.unmasked[:,pixel]] = fitSpectrum

        # update mask on input data
        self.applyMask()

    def plot_example_fit(self,indep=1,pixel=0,
                         xlabel='$T_{\mathrm{eff}}$ - median($T_{\mathrm{eff}}$) (K)',figsize=(12,8)):
        """
        Show a two-dimensional representation of the fit at a given pixel.

        indep:   Column of matrix from makeMatrix corresponding to the
                 independent variable to plot against.
        pixel:   Pixel at which to plot the fit.
        xlabel:  Label for the x-axis of the plot.
        """
        # Create figure
        plt.figure(figsize=figsize)
        ax=plt.subplot2grid((3,1),(0,0),rowspan=2)
        # Find independent value
        indeps = self.makeMatrix(pixel)
        fitresult = np.dot(indeps,self.fitCoeffs[pixel].T)
        indep = np.array(np.reshape(indeps[:,indep],len(fitresult)))[0]
        sortd = indep.argsort()
        fitindep = np.arange(np.floor((min(indep)-100)/100.)*100,np.ceil((max(indep)+100)/100.)*100,100)
        fit = np.dot(np.matrix([np.ones(len(fitindep)),fitindep,fitindep**2]).T,self.fitCoeffs[pixel].T)
        c = plt.get_cmap('viridis')(np.linspace(0.6, 1, 1))[0]
        unmasked = np.where(self.spectra[:,pixel].mask==False)
        # Plot a fit line and errorbar points of data
        plt.plot(fitindep,fit,lw=3,color='k',
                 label='$f(s,T_{\mathrm{eff}}$)')
        plt.plot(indep,self.spectra[:,pixel][unmasked],'o',color=c,
                 markersize=10,markeredgecolor='w',markeredgewidth=1.5)
        plt.ylabel('stellar flux $F_p(s)$',fontsize=20)
        plt.xticks(np.arange(np.floor((min(indep)-100)/100.)*100+100,
                             np.ceil((max(indep)+100)/100.)*100,100),['']*10)
        plt.ylim(0.6,1.1)
        plt.xlim(np.floor((min(indep)-100)/100.)*100+100,
                 np.ceil((max(indep)+100)/100.)*100-100)
        plt.yticks(np.arange(0.7,1.1,0.1),np.arange(0.7,1.1,0.1).astype(str),
                   fontsize=20)
        yminorlocator = MultipleLocator(0.05)
        ax.yaxis.set_minor_locator(yminorlocator)
        xminorlocator = MultipleLocator(50)
        ax.xaxis.set_minor_locator(xminorlocator)
        plt.tick_params(which='both', width=2)
        plt.tick_params(which='major',length=5)
        plt.tick_params(which='minor',length=3)
        plt.legend(loc='best',frameon=False)
        # Plot residuals of the fit
        ax=plt.subplot2grid((3,1),(2,0))
        plt.axhline(0,lw=3,color='k')
        #for i in range(len(indep[sortd][unmasked])):
        #plt.plot(indep,self.residuals[:,pixel][unmasked],'o',color=c,
        #         markersize=10,markeredgecolor='w',markeredgewidth=1.5)
        plt.errorbar(indep,self.residuals[:,pixel][unmasked],yerr=self.spectra_errs[:,pixel][unmasked],fmt='o',color=c,markersize=8,elinewidth=2,markeredgecolor='w',markeredgewidth=1.5)
        plt.ylabel('residuals $\delta_p(s)$ ',fontsize=20)
        plt.xlabel(xlabel,fontsize=20)
        plt.xticks(fontsize=16)
        plt.ylim(-0.05,0.05)
        plt.xlim(np.floor((min(indep)-100)/100.)*100+100,
                 np.ceil((max(indep)+100)/100.)*100-100)
        #plt.yticks(np.arange(-0.1,0.15,0.1),
        #           np.arange(-0.1,0.15,0.1).astype(str))
        yminorlocator = MultipleLocator(0.01)
        ax.yaxis.set_minor_locator(yminorlocator)
        xminorlocator = MultipleLocator(50)
        ax.xaxis.set_minor_locator(xminorlocator)
        plt.tick_params(which='both', width=2)
        plt.tick_params(which='major',length=5)
        plt.tick_params(which='minor',length=3)
        plt.subplots_adjust(hspace=0)

    def fitStatistic(self):
        """
        Adds to fit object chi squared and reduced chi squared properties.

        """
        self.fitChiSquared = np.ma.sum((self.spectra-self.fitSpectra)**2/self.spectra_errs**2,axis=0)
        # Calculate degrees of freedom
        if isinstance(self.fitCoeffs.mask,np.ndarray):
            dof = self.numberStars() - np.sum(self.fitCoeffs.mask==False,axis=1) - 1
        else:
            dof = self.numberStars() - self.numparams - 1
        self.fitReducedChi = self.fitChiSquared/dof
    
    def findResiduals(self,minStarNum='default',gen=True,coeffs=None,matrix='default'):
        """
        Calculate residuals from polynomial fits.
        
        minStarNum:   (optional) number of stars required to perform fit 
                      (default:'default' which sets minStarNum to the number 
                       of fit parameters plus one)
        
        """
        if gen:
            self.multiFit(minStarNum=minStarNum,coeffs=coeffs,matrix=matrix)
            self.residuals = self.spectra - self.fitSpectra 
            np.save(self.name+'/fitcoeffs.npy',self.fitCoeffs.data)
            np.save(self.name+'/fitcoeffmask.npy',self.fitCoeffs.mask)
            np.save(self.name+'/fitcoefferrs.npy',self.fitCoeffErrs.data)
            np.save(self.name+'/fitspectra.npy',self.fitSpectra.data)
            np.save(self.name+'/residuals.npy',self.residuals.data)
            np.save(self.name+'/mask.npy',self.masked)
        if not gen:
            self.testM = self.makeMatrix(0)
            self.minStarNum = self.testM.shape[1]+1
            fmask = np.load(self.name+'/fitcoeffmask.npy')
            self.fitCoeffs = np.ma.masked_array(np.load(self.name+'/fitcoeffs.npy'),mask=fmask)
            self.fitCoeffErrs = np.ma.masked_array(np.load(self.name+'/fitcoefferrs.npy'),mask=fmask)
            self.fitSpectra = np.ma.masked_array(np.load(self.name+'/fitspectra.npy'),mask=self.masked)
            self.residuals = np.ma.masked_array(np.load(self.name+'/residuals.npy'),mask=self.masked)
            
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
    '''

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
    '''

    def findCorrection(self,cov=None,median=True,numpix=10.,frac=None,
                       savename='pickles/correction_factor.pkl',tol=0.005):
        """
        Calculates the diagonal of a square matrix and smooths it
        either over a fraction of the data or a number of elements,
        where number of elements takes precedence if both are set.

        cov:      Square matrix. If unspecified, calculate from residuals and 
                  uncertainties
        median:   If true, returns smoothed median, not raw diagonal
        numpix:   Number of elements to smooth over
        frac:     Fraction of data to smooth over
        tol:      Acceptable distance from 0, if greater than this, smooth to 
                  adjacent values when median is True

        Returns the diagonal of a covariance matrix
        """
        if isinstance(cov,(list,np.ndarray)):
            self.cov=cov
        if not isinstance(cov,(list,np.ndarray)):
            arr = self.residuals.T/self.spectra_errs.T
            cov = np.ma.cov(arr)
            self.cov=cov
        diagonal = np.ma.diag(cov)
        if median:
            median = smoothMedian(diagonal,frac=frac,numpix=float(numpix))
            offtol = np.where(np.fabs(median)<tol)[0]
            if offtol.shape > 0:
                median[offtol] = median[offtol-1]
            acs.pklwrite(savename,median)
            return median
        elif not median:
            acs.pklwrite(savename,diagonal)
            return diagonal
                    

    def pixelEMPCA(self,randomSeed=1,nvecs=5,deltR2=0,varfunc=np.ma.var,correction=None,savename=None,gen=True,numpix=None,weight=True):
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
            self.applyMask()
            self.nvecs = nvecs
            self.deltR2 = deltR2
            # Find pixels with enough stars to do EMPCA
            self.goodPixels=([i for i in range(aspcappix) if np.sum(self.residuals[:,i].mask) < self.residuals.shape[0]-self.minStarNum],)
            self.empcaResiduals = self.residuals.T[self.goodPixels].T
            
            # Calculate weights that just mask missing elements
            unmasked = (self.empcaResiduals.mask==False)
            errorWeights = unmasked.astype(float)
            if weight:
                errorWeights[unmasked] = 1./((self.spectra_errs.T[self.goodPixels].T[unmasked])**2)
            self.empcaModelWeight = empca(self.empcaResiduals.data,weights=errorWeights,
                                          nvec=self.nvecs,deltR2=self.deltR2,
                                          randseed=randomSeed,varfunc=varfunc)    

            # Calculate eigenvalues
            self.empcaModelWeight.eigvals = np.zeros(len(self.empcaModelWeight.eigvec))
            for e in range(len(self.empcaModelWeight.eigvec)):
                self.empcaModelWeight.eigvals[e] = self.empcaModelWeight.eigval(e+1)

            # Find R2 and R2noise for this model, and resize eigenvectors appropriately
            self.setR2(self.empcaModelWeight)
            self.setDeltaR2(self.empcaModelWeight)
            self.setR2noise(self.empcaModelWeight)
            self.resizePixelEigvec(self.empcaModelWeight)

            self.uncorrectUncertainty(correction=correction)
            self.applyMask()
            self.smallModel = smallEMPCA(self.empcaModelWeight,correction=correction,savename=self.name+'/'+savename)
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
            R2Array[vec] = model.R2(vec)
        # Add R2 array to model
        model.R2Array = R2Array

    def setDeltaR2(self,model):
        """
        """
        self.setR2(model)
        model.DeltaR2 = (np.roll(model.R2Array,-1)-model.R2Array)[:-1]

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
        model.Vdata = model._unmasked_data_var
        # Calculate data noise
        model.Vnoise = np.mean(1./(model.weights[model.weights!=0]))
        # Calculate R2noise
        model.R2noise = 1.-(model.Vnoise/model.Vdata)

