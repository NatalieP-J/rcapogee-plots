import numpy as np
from scipy.interpolate import interp1d
from spectralspace.analysis.empca_residuals import *

def getarrays(model):
    """
    Read out arrays associated with a model holding EMPCA information.

    model:     EMPCA model object

    Returns updated model object.
    """
    if model.savename:
        savename = model.savename.split('/')
        parts = [i for i in savename if 'seed' not in i]
        savename = ('/').join(parts)
        arc = np.load('{0}_data.npz'.format(savename))

        model.eigval = arc['eigval']
        model.eigvec = np.ma.masked_array(arc['eigvec'],
                                         mask=arc['eigvecmask'])
        model.coeff = arc['coeff']
    return model

def reconstruct_EMPCA_data(direc,model,minStarNum=5):
    """
    Reconstruct the data used in the EMPCA analysis (with appropriate weights)

    direc:        Directory where model files are stored
    model:        Model object to which attributes will be assigned
    minstarNum:   Minimum number of stars required to perform EMPCA on a
                  given pixel, default 5

    Returns updated model object
    """
    model = getarrays(model)
    # Find parent directory where basic spectral information is housed
    skimbm = direc.split('/')
    parentdirec = '/'.join(skimbm[:-1])
    # Load in uncertainties
    spectra_errs = np.load('{0}/spectra_errs.npy'.format(parentdirec))
    spectra = np.load('{0}/spectra.npy'.format(parentdirec))
    # Load in mask
    mask = np.load('{0}/mask.npy'.format(direc))
    # Load in fit residuals
    residuals = np.load('{0}/residuals.npy'.format(direc))
    fitspec = np.load('{0}/fitspectra.npy'.format(direc))
    residuals = np.ma.masked_array(residuals,mask=mask)
    spectra_errs = np.ma.masked_array(spectra_errs,mask=mask)
    # Find pixels with enough stars to do EMPCA
    goodPixels=([i for i in range(aspcappix) if np.sum(residuals[:,i].mask) < residuals.shape[0]-minStarNum],)
    empcaResiduals = residuals.T[goodPixels].T

    # Calculate weights that just mask missing elements
    unmasked = (empcaResiduals.mask==False)
    errorWeights = unmasked.astype(float)
    errorWeights[unmasked] = 1./((spectra_errs.T[goodPixels].T[unmasked])**2)
    # Assign attributes to the model
    model.residuals = residuals
    model.data = empcaResiduals
    model.weights = errorWeights
    model.fitspec = fitspec
    model.spectra = spectra
    model.spectra_errs=spectra_errs
    model.mask = mask
    return model


def consth(n,model,N,D,scale=1):
    """
    Assume that all chemical space cells have the same size and calculate
    that size from measurement uncertainties.

    n:          Number of eigenvectors for which to calculate cell size
    model:      EMPCA model object that contains eigenvalues
    N:          Number of measurements in the data set (i.e. number of stars)
    D:          Number of dimensions in each measurement (i.e. pixels)
    scale:      Factor by which to scale resulting cell size.

    Returns an array of length n filled with the chemical space cell size.
    """
    hs = np.ones(n)
    hs = np.sqrt((1./len(model.weights[model.weights!=0]))*np.sum(1./model.weights[model.weights!=0]))*hs
    return scale*hs

def pessimh(n,model,N,D,cvc=None):
    """
    Assume that all chemical space cells have the same size and calculate
    that size using the eigenvalue of the intersection point between R^2 and
    R^2_noise.
    n:          Number of eigenvectors for which to calculate cell size
    model:      EMPCA model object that contains eigenvalues
    N:          Number of measurements in the data set (i.e. number of stars)
    D:          Number of dimensions in each measurement (i.e. pixels)
    cvc:        User-specified point of intersection

    Returns an array of length n filled with the chemical space cell size.
    """
    # If interesection point not given, calculate
    if not cvc:
        crossvec = np.where(model.R2Array > model.R2noise)
        # If intersection exists, use it
        if crossvec[0] != []:
            crossvec = crossvec[0][0] - 1
            if crossvec < 0:
                crossvec = 0
            # Scale value appropriately
            if D < N:
                hs = np.ones(n)*np.sqrt(D*model.eigval[crossvec])
            elif N < D:
                hs = np.ones(n)*np.sqrt((N**2/D)*model.eigval[crossvec])

        # If intersection does not exist, use the max number of eigenvectors
        # and scale appropriately
        else:
            if D < N:
                hs = np.ones(n)*np.sqrt(D*model.eigval[n-1])
            elif N < D:
                hs = np.ones(n)*np.sqrt((N**2/D)*model.eigval[n-1])

    # If intersection point given, use that
    elif cvc:
        crossvec=cvc-1
        lamnoise2 = np.interp(crossvec,np.arange(len(model.eigval)),model.eigval)
        # Scale value appropriately
        if D < N:
                hs = np.ones(n)*np.sqrt(D*lamnoise2)
        elif N < D:
            hs = np.ones(n)*np.sqrt((N**2/D)*lamnoise2)
    return hs

def calculate_Ncells(direc,model,modelname,N=None,D=None,denom=consth,
                     generate=False,**kwargs):
    """
    Calculate the number of chemical space cells as a function of the number of eigenvectors.

    direc:      Directory where model files are stored
    model:      EMPCA model object that contains eigenvalues
    modelname:  Name of the model object for saving files
    N:          Number of measurements in the data set (i.e. number of stars)
    D:          Number of dimensions in each measurement (i.e. pixels)
    denom:      If function given, use that to calculate denominator.
                If constant or array given, use that.
    generate:   If False, read Ncells object from file. If True generate Ncell
                object from scratch
    **kwargs:   Keyword arguments for denominator function

    Returns Ncells object as a scipy interp1d object.
    """
    # Create output object file name
    if isinstance(denom,(int,float)):
        fname = 'Ncells_{0}_{1}.npy'.format(modelname,denom)
    if isinstance(denom,(list,np.ndarray)):
        fname = 'Ncells_{0}_listwith{1}.npy'.format(modelname,denom[0])
    if callable(denom):
        fname = 'Ncells_{0}_{1}.npy'.format(modelname,denom.__name__)
    # Read object if generate is False and file exists
    if os.path.isfile('{0}/{1}'.format(direc,fname)) and not generate:
        Ncells = np.load('{0}/{1}'.format(direc,fname))
    # Create object if generate is True or file doesn't exist
    elif not os.path.isfile('{0}/{1}'.format(direc,fname)) or generate:
        model = reconstruct_EMPCA_data(direc,model,minStarNum=5)
        numeig = len(model.eigval)
        # Determine array size if dimensions not given
        if not N or not D:
            D = model.data.shape[1]
            N = model.data.shape[0]
        # Create array of denominator values
        if isinstance(denom,(int,float)):
            denomarr = denom*np.ones(numeig)
        if isinstance(denom,list):
            denomarr = np.array(denom)
        if callable(denom):
            denomarr = denom(numeig,model,N,D,**kwargs)
        try:
            shapetest = denomarr*np.ones(numeig)
        except ValueError as e:
            print('Input denominator array has invalid shape, should be {0}'.format(len(model.eigval)))
            print(e)
            return None
        denom=denomarr
        # Constrain that cell size is never larger than span of chemical space
        newdenom = np.copy(denom)
        cond = np.sqrt(N*model.eigval) < denom
        newdenom[cond] = np.sqrt(N*model.eigval[cond])
        denom = newdenom

        def calcNcells(n,denom,D,N):
            """
            Calculate the number of the chemical space cells for the space
            spanned by first n eigenvectors.

            n:      Index of the eigenvector to use
            denom:  Array of denominators for all eigenvectors
            D:      Number of dimensions in each measurement (i.e. pixels)
            N:      Number of measurements in the data set (number of stars)

            Return number of cells for a space defined by first n principal
            components
            """
            if D < N:
                return np.prod(np.sqrt(D*model.eigval[:n+1]))/np.prod(denom[:n+1])
            elif N < D:
                return np.prod(np.sqrt((N*model.eigval)[:n+1]))/(np.prod((denom)[:n+1]))
        # Find the number of cells for each successive principal component
        Ncells = np.zeros(numeig)
        for n in range(len(model.eigval)):
            Ncells[n] = calcNcells(n,denom,D,N)
        np.save('{0}/{1}'.format(direc,fname),Ncells)
    # Linear interpolation
    Ncells = interp1d(np.arange(len(Ncells))+1,Ncells)
    return Ncells
