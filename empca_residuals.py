import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import statsmodels.nonparametric.smoothers_lowess as sm
import access_spectrum as acs
from empca import empca
from mask_data import mask,maskFilter
from data_access import *
import access_spectrum as acs 
import os

font = {'family': 'serif',
        'weight': 'normal',
        'size'  :  20
}

matplotlib.rc('font',**font)
plt.ion()

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


class smallEMPCA(object):
    """
    
    Class to contain crucial EMPCA-related objects.

    """
    def __init__(self,model,correction=None):
        """
        Set all relevant data for the EMPCA.

        model:        EMPCA model object
        correction:   information about correction used on data
        """
        self.R2Array = model.R2Array
        self.R2noise = model.R2noise
        self.mad = model.mad
        self.eigvec = model.eigvec
        self.coeff = model.coeff
        self.correction = correction
        self.eigval = model.eigval


class empca_residuals(mask):
    """
    Contains functions to find polynomial fits.
    
    """
    def __init__(self,dataSource,sampleType,maskFilter,ask=True,degree=2):
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
        mask.__init__(self,dataSource,sampleType,maskFilter,ask=ask)
        self.degree = degree
        self.noncrossInds = ([0,1,2,3,4,7,9])
        self.crossInds = ([5,6,8],)
        self.polynomial = PolynomialFeatures(degree=degree)
        self.testM = self.makeMatrix(0)

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
                         xlabel='$T_{\mathrm{eff}}$ - median($T_{\mathrm{eff}}$) [K]'):
        """
        Show a two-dimensional representation of the fit at a given pixel.

        indep:   Column of matrix from makeMatrix corresponding to the
                 independent variable to plot against.
        pixel:   Pixel at which to plot the fit.
        xlabel:  Label for the x-axis of the plot.
        """
        # Create figure
        plt.figure(figsize=(12,8))
        plt.subplot2grid((3,1),(0,0),rowspan=2)
        # Find and sort independent value
        indeps = self.makeMatrix(pixel)
        fitresult = np.dot(indeps,self.fitCoeffs[pixel].T)
        indep = np.array(np.reshape(indeps[:,indep],len(fitresult)))[0]
        sortd = indep.argsort()
        fitindep = np.arange(np.floor((min(indep)-100)/100.)*100,np.ceil((max(indep)+100)/100.)*100,100)
        fit = np.dot(np.matrix([np.ones(len(fitindep)),fitindep,fitindep**2]).T,self.fitCoeffs[pixel].T)
        colors = plt.get_cmap('plasma')(np.linspace(0, 0.8, len(indep)))
        unmasked = np.where(self.spectra[:,pixel][sortd].mask==False)
        # Plot a fit line and errorbar points of data
        plt.plot(fitindep,fit,lw=3,color='k',
                 label='$f(s,T_{\mathrm{eff}}$)')
        print len(indep[sortd][unmasked])
        print sortd
        print unmasked
        for i in range(len(indep[sortd][unmasked])):
            plt.errorbar(indep[sortd][unmasked][i],
                         self.spectra[:,pixel][sortd][unmasked][i],
                         markerfacecolor=colors[i],fmt='o',
                         markeredgecolor='w',markeredgewidth=1.5,
                         ecolor = colors[i],capsize=5,elinewidth=3,
                         markersize=8,
                         yerr=self.spectra_errs[:,pixel][sortd][unmasked][i])
        plt.ylabel('stellar flux $F_p(s)$',fontsize=20)
        plt.xticks([])
        plt.ylim(0.6,1.1)
        plt.xlim(np.floor((min(indep)-100)/100.)*100+100,
                 np.ceil((max(indep)+100)/100.)*100-100)
        plt.yticks(np.arange(0.7,1.1,0.1),np.arange(0.7,1.1,0.1).astype(str))
        plt.legend(loc='best',frameon=False)
        # Plot residuals of the fit
        plt.subplot2grid((3,1),(2,0))
        plt.axhline(0,lw=3,color='k')
        for i in range(len(indep[sortd][unmasked])):
            plt.errorbar(indep[sortd][unmasked][i],
                         self.residuals[:,pixel][sortd][unmasked][i],
                         markerfacecolor=colors[i],fmt='o',
                         markeredgecolor='w',markeredgewidth=1.5,
                         ecolor = colors[i],capsize=5,elinewidth=3,
                         markersize=8,
                         yerr=self.spectra_errs[:,pixel][sortd][unmasked][i])
        plt.ylabel('residuals $\delta_p(s)$ ',fontsize=20)
        plt.xlabel(xlabel,fontsize=20)
        plt.ylim(-0.1,0.1)
        plt.xlim(np.floor((min(indep)-100)/100.)*100+100,
                 np.ceil((max(indep)+100)/100.)*100-100)
        plt.yticks(np.arange(-0.1,0.2,0.1),
                   np.arange(-0.1,0.2,0.1).astype(str))
        plt.subplots_adjust(hspace=0)

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
        if not cov:
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
            
    def findEigval(self,eigvec,matrix):
        scaledEigvec = np.dot(np.dot(matrix.T,matrix),eigvec)
        print scaledEigvec/eigvec, 
        return scaledEigvec/eigvec[0]

    def pixelEMPCA(self,randomSeed=1,nvecs=5,deltR2=0,mad=False,correction=None,savename=None,gen=True,numpix=None,weight=True):
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
            self.mad = mad
            self.nvecs = nvecs
            self.deltR2 = deltR2
            # Find pixels with enough stars to do EMPCA
            self.goodPixels=([i for i in range(aspcappix) if np.sum(self.residuals[:,i].mask) < self.residuals.shape[0]-self.minStarNum],)
            empcaResiduals = self.residuals.T[self.goodPixels].T
            
            # Calculate weights that just mask missing elements
            unmasked = (empcaResiduals.mask==False)
            errorWeights = unmasked.astype(float)
            if weight:
                errorWeights[unmasked] = 1./((self.spectra_errs.T[self.goodPixels].T[unmasked])**2)
            self.empcaModelWeight = empca(empcaResiduals.data,weights=errorWeights,
                                          nvec=self.nvecs,deltR2=self.deltR2,
                                          mad=self.mad,randseed=randomSeed)    

            self.empcaModelWeight.mad = self.mad
            # Calculate eigenvalues
            self.empcaModelWeight.eigval = np.zeros(len(self.empcaModelWeight.eigvec))
            for e in range(len(self.empcaModelWeight.eigvec)):
                eigval = self.findEigval(self.empcaModelWeight.eigvec[e],
                                         empcaResiduals.data)
                self.empcaModelWeight.eigval[e] = eigval
                
            # Find R2 and R2noise for this model, and resize eigenvectors appropriately
            self.setR2(self.empcaModelWeight)
            self.setDeltaR2(self.empcaModelWeight)
            self.setR2noise(self.empcaModelWeight)
            self.resizePixelEigvec(self.empcaModelWeight)

            self.uncorrectUncertainty(correction=correction)
            self.applyMask()
            self.smallModel = smallEMPCA(self.empcaModelWeight,correction=correction)
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

